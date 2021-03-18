import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, p_dim, c_dim):
        super(Generator, self).__init__()
        self.p_dim = p_dim
        self.c_dim = c_dim
        convex_layer_weights = torch.zeros((self.p_dim, self.c_dim))
        self.convex_layer_weights = nn.Parameter(convex_layer_weights)
        nn.init.normal_(self.convex_layer_weights, mean=0.0, std=0.02)

    def forward(self, points, plane_m, convex_mask=None, is_training=False):
        #level 1
        h1 = torch.matmul(points, plane_m)
        h1 = torch.clamp(h1, min=0)

        #level 2
        h2 = torch.matmul(h1, (self.convex_layer_weights>0.01).float())

        #level 3
        if convex_mask is None:
            h3 = torch.min(h2, dim=2, keepdim=True)[0]
        else:
            h3 = torch.min(h2+convex_mask, dim=2, keepdim=True)[0]

        return h2,h3


class ResnetBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ResnetBlock, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        if self.dim_in == self.dim_out:
            self.conv_1 = nn.Conv2d(self.dim_in, self.dim_out, 3, stride=1, padding=1, bias=False)
            self.conv_2 = nn.Conv2d(self.dim_out, self.dim_out, 3, stride=1, padding=1, bias=False)
            nn.init.xavier_uniform_(self.conv_1.weight)
            nn.init.xavier_uniform_(self.conv_2.weight)
        else:
            self.conv_1 = nn.Conv2d(self.dim_in, self.dim_out, 3, stride=2, padding=1, bias=False)
            self.conv_2 = nn.Conv2d(self.dim_out, self.dim_out, 3, stride=1, padding=1, bias=False)
            self.conv_s = nn.Conv2d(self.dim_in, self.dim_out, 1, stride=2, padding=0, bias=False)
            nn.init.xavier_uniform_(self.conv_1.weight)
            nn.init.xavier_uniform_(self.conv_2.weight)
            nn.init.xavier_uniform_(self.conv_s.weight)

    def forward(self, input, is_training=False):
        if self.dim_in == self.dim_out:
            output = self.conv_1(input)
            output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
            output = self.conv_2(output)
            output = output+input
            output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
        else:
            output = self.conv_1(input)
            output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
            output = self.conv_2(output)
            input_ = self.conv_s(input)
            output = output+input_
            output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
        return output


class ImageEncoder(nn.Module):
    def __init__(self, img_ef_dim, z_dim):
        super(ImageEncoder, self).__init__()
        self.img_ef_dim = img_ef_dim
        self.z_dim = z_dim
        self.conv_0 = nn.Conv2d(1, self.img_ef_dim, 7, stride=2, padding=3, bias=False)
        self.res_1 = ResnetBlock(self.img_ef_dim, self.img_ef_dim)
        self.res_2 = ResnetBlock(self.img_ef_dim, self.img_ef_dim)
        self.res_3 = ResnetBlock(self.img_ef_dim, self.img_ef_dim*2)
        self.res_4 = ResnetBlock(self.img_ef_dim*2, self.img_ef_dim*2)
        self.res_5 = ResnetBlock(self.img_ef_dim*2, self.img_ef_dim*4)
        self.res_6 = ResnetBlock(self.img_ef_dim*4, self.img_ef_dim*4)
        self.res_7 = ResnetBlock(self.img_ef_dim*4, self.img_ef_dim*8)
        self.res_8 = ResnetBlock(self.img_ef_dim*8, self.img_ef_dim*8)
        self.conv_9 = nn.Conv2d(self.img_ef_dim*8, self.img_ef_dim*16, 4, stride=2, padding=1, bias=True)
        self.conv_10 = nn.Conv2d(self.img_ef_dim*16, self.img_ef_dim*16, 4, stride=1, padding=0, bias=True)
        self.linear_1 = nn.Linear(self.img_ef_dim*16, self.img_ef_dim*16, bias=True)
        self.linear_2 = nn.Linear(self.img_ef_dim*16, self.img_ef_dim*16, bias=True)
        self.linear_3 = nn.Linear(self.img_ef_dim*16, self.img_ef_dim*16, bias=True)
        self.linear_4 = nn.Linear(self.img_ef_dim*16, self.z_dim, bias=True)
        nn.init.xavier_uniform_(self.conv_0.weight)
        nn.init.xavier_uniform_(self.conv_9.weight)
        nn.init.constant_(self.conv_9.bias,0)
        nn.init.xavier_uniform_(self.conv_10.weight)
        nn.init.constant_(self.conv_10.bias,0)
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.constant_(self.linear_1.bias,0)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.constant_(self.linear_2.bias,0)
        nn.init.xavier_uniform_(self.linear_3.weight)
        nn.init.constant_(self.linear_3.bias,0)
        nn.init.xavier_uniform_(self.linear_4.weight)
        nn.init.constant_(self.linear_4.bias,0)

    def forward(self, view, is_training=False):
        layer_0 = self.conv_0(1-view)
        layer_0 = F.leaky_relu(layer_0, negative_slope=0.01, inplace=True)

        layer_1 = self.res_1(layer_0, is_training=is_training)
        layer_2 = self.res_2(layer_1, is_training=is_training)
        
        layer_3 = self.res_3(layer_2, is_training=is_training)
        layer_4 = self.res_4(layer_3, is_training=is_training)
        
        layer_5 = self.res_5(layer_4, is_training=is_training)
        layer_6 = self.res_6(layer_5, is_training=is_training)
        
        layer_7 = self.res_7(layer_6, is_training=is_training)
        layer_8 = self.res_8(layer_7, is_training=is_training)
        
        layer_9 = self.conv_9(layer_8)
        layer_9 = F.leaky_relu(layer_9, negative_slope=0.01, inplace=True)
        
        layer_10 = self.conv_10(layer_9)
        layer_10 = layer_10.view(-1,self.img_ef_dim*16)
        layer_10 = F.leaky_relu(layer_10, negative_slope=0.01, inplace=True)

        l1 = self.linear_1(layer_10)
        l1 = F.leaky_relu(l1, negative_slope=0.01, inplace=True)

        l2 = self.linear_2(l1)
        l2 = F.leaky_relu(l2, negative_slope=0.01, inplace=True)

        l3 = self.linear_3(l2)
        l3 = F.leaky_relu(l3, negative_slope=0.01, inplace=True)

        l4 = self.linear_4(l3)
        l4 = torch.sigmoid(l4)

        return l4


class Decoder(nn.Module):
    def __init__(self, ef_dim, p_dim):
        super(Decoder, self).__init__()
        self.ef_dim = ef_dim
        self.p_dim = p_dim
        self.linear_1 = nn.Linear(self.ef_dim*8, self.ef_dim*16, bias=True)
        self.linear_2 = nn.Linear(self.ef_dim*16, self.ef_dim*32, bias=True)
        self.linear_3 = nn.Linear(self.ef_dim*32, self.ef_dim*64, bias=True)
        self.linear_4 = nn.Linear(self.ef_dim*64, self.p_dim*4, bias=True)
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.constant_(self.linear_1.bias,0)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.constant_(self.linear_2.bias,0)
        nn.init.xavier_uniform_(self.linear_3.weight)
        nn.init.constant_(self.linear_3.bias,0)
        nn.init.xavier_uniform_(self.linear_4.weight)
        nn.init.constant_(self.linear_4.bias,0)

    def forward(self, inputs, is_training=False):
        l1 = self.linear_1(inputs)
        l1 = F.leaky_relu(l1, negative_slope=0.01, inplace=True)

        l2 = self.linear_2(l1)
        l2 = F.leaky_relu(l2, negative_slope=0.01, inplace=True)

        l3 = self.linear_3(l2)
        l3 = F.leaky_relu(l3, negative_slope=0.01, inplace=True)

        l4 = self.linear_4(l3)
        l4 = l4.view(-1, 4, self.p_dim)

        return l4


class BSPNet(nn.Module):
    def __init__(self, ef_dim, p_dim, c_dim, img_ef_dim, z_dim):
        super(BSPNet, self).__init__()
        self.ef_dim = ef_dim
        self.p_dim = p_dim
        self.c_dim = c_dim
        self.img_ef_dim = img_ef_dim
        self.z_dim = z_dim
        self.img_encoder = ImageEncoder(self.img_ef_dim, self.z_dim)
        self.decoder = Decoder(self.ef_dim, self.p_dim)
        self.generator = Generator(self.p_dim, self.c_dim)

    def forward(self, inputs, z_vector, plane_m, point_coord, convex_mask=None, is_training=False):
        if is_training:
            z_vector = self.img_encoder(inputs, is_training=is_training)
            plane_m = None
            net_out_convexes = None
            net_out = None
        else:
            if inputs is not None:
                z_vector = self.img_encoder(inputs, is_training=is_training)
            if z_vector is not None:
                plane_m = self.decoder(z_vector, is_training=is_training)
            if point_coord is not None:
                net_out_convexes, net_out = self.generator(point_coord, plane_m, convex_mask=convex_mask, is_training=is_training)
            else:
                net_out_convexes = None
                net_out = None

        return z_vector, plane_m, net_out_convexes, net_out
