import torch
from buildingblocks import DoubleConv, ExtResNetBlock, create_encoders, create_decoders

def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]

class DefNet(torch.nn.Module):
    def __init__(self, inshape, in_channels = 1, out_channels = 2, verbose = False):
        super(DefNet, self).__init__()

        f_maps = 64
        layer_order = "gcr"
        num_groups = 8
        num_levels = 4
        conv_kernel_size = 3
        pool_kernel_size = 2
        conv_padding = 1
        basic_module = DoubleConv
        #res_module = ExtResNetBlock
        testing = False
        self.in_channels = in_channels

        self.testing = testing
        self.verbose = verbose

        #make the feature maps?
        f_maps = number_of_features_per_level(f_maps, num_levels = num_levels)

        #create encoder path
        self.encoders = create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups, pool_kernel_size)
        #self.encoders = create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups, pool_kernel_size)

        #create decoder paths
        self.def_decoders = create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups, upsample=True)
        #self.def_decoders = create_decoders(f_maps, res_module, conv_kernel_size, conv_padding, layer_order, num_groups, upsample=True)

        #make names
        self.encoders.name = "Shared Encoders"
        self.def_decoders.name = "Deformation Decoders"

        #make a layer for the flow
        self.flow = torch.nn.Conv3d(f_maps[0], len(inshape), 1)
        # init flow layer with small weights and bias
        self.flow.weight = torch.nn.Parameter(torch.distributions.normal.Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = torch.nn.Parameter(torch.zeros(self.flow.bias.shape))

        #make a transform function for the flow
        self.transformer = MyTransformer(inshape)
        return

    def forward(self, combo_input):
        #x channels = [mask, image]
        #x shape = [batch size, channels, size, size, size]
        #combo_input = mask, image, atlas mask, atlas image

        x = combo_input
        
        if self.verbose:
            print("network input:", x.shape)

        '''encoders'''
        encoder_features = [] #the list of features for the decoders to use
        e = -1
        for encoder in self.encoders:
            e = e + 1
            if self.verbose:
                print("encoder", e)
                print("\tencoder input:", x.shape)
            x = encoder(x) #run the input through the encoders
            if self.verbose:
                print("\tencoder output:", x.shape)
            encoder_features.insert(0, x) #flip the order so it matches the order of the decoders
        encoder_features = encoder_features[1:] #remove the final encoder output from the list. unsure why.

        x_def = x

        '''def decoders'''
        d = -1
        for i in range(len(self.def_decoders)):
            d = d + 1
            decoder = self.def_decoders[i]
            encoder_feature = encoder_features[i]
            if self.verbose:
                print("decoder", d)
                print("\tdef decoder inputs:", encoder_feature.shape, x_def.shape)
            x_def = decoder(encoder_feature, x_def) #run the input through the decoders, using the encoder feature info
            if self.verbose:
                print("\tdef decoder output:", x_def.shape)

        '''flow layer'''
        flow_field = self.flow(x_def)
        if self.verbose:
            print("flow final:", flow_field.shape)

        to_transform = combo_input[:,self.in_channels-1,:,:,:].unsqueeze(1)

        y_source = self.transformer(to_transform, flow_field)

        if torch.isnan(flow_field.min()):
            dsllkfjdslfsjdf #just for testing
        
        return y_source, flow_field

class MyTransformer(torch.nn.Module):
    def __init__(self, size, mode='bilinear'):
        super().__init__()
        xx = torch.arange(0,size[0])
        yy = torch.arange(0,size[1])
        zz = torch.arange(0,size[2])
        XYZ = torch.meshgrid(xx,yy,zz, indexing = "ij")
        grid = torch.stack(XYZ)
        self.register_buffer('grid', grid)
        self.mode = mode
    def forward(self, src, flow):
        # new locations
        #print(self.grid.shape, flow.shape)
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return torch.nn.functional.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
