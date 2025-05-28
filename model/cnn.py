from model.dependencies import *

class cnn_embed(nn.Module):
    '''
    input: retina
    output: cnn embedding of retina
    '''
    def __init__(
        self,
        cnn_hidden_lyrs: list = [16, 32],
        lin_hidden_lyrs: list = [32, 64],
        input_img_shape: tuple = (60, 80) # HW
    ):
        ### 0)
        super().__init__()
        
        ### 1) CNN LAYERS
        self.cnn_lyrs = []
        cnn_hidden_lyr_prev = 1 ## TBD iINITIAL CHANNEL SET TO 1
        for cnn_hidden_lyr in cnn_hidden_lyrs:
            conv = nn.Conv2d(in_channels = cnn_hidden_lyr_prev,
                            out_channels = cnn_hidden_lyr,
                            kernel_size=3,
                            stride=1,
                            padding=0)
            self.cnn_lyrs += [conv, nn.ReLU(), nn.MaxPool2d(2,2)]
            cnn_hidden_lyr_prev = cnn_hidden_lyr

        ### 2) LINEAR LAYERS
        self.lin_lyrs = [nn.Flatten()]
        if lin_hidden_lyrs == []:
            pass 
        else:
            output_img_shape = self.get_shape(self.cnn_lyrs, (1,1, *input_img_shape)) ## TBD BATCH
            lin_hidden_lyr_prev = math.prod(output_img_shape)
            for lin_hidden_lyr in lin_hidden_lyrs[:-1]:
                self.lin_lyrs += [
                    nn.Linear(lin_hidden_lyr_prev, lin_hidden_lyr),
                    nn.ReLU()
                ]
                lin_hidden_lyr_prev = lin_hidden_lyr

            self.lin_lyrs.append(nn.Linear(lin_hidden_lyr_prev, lin_hidden_lyrs[-1])) ##TBD CHECK
        
        self.cnn_model = nn.Sequential(*self.cnn_lyrs)
        self.lin_model = nn.Sequential(*self.lin_lyrs)

    def forward(self, input_img, only_output=True):
        inter_img = self.cnn_model(input_img) 
        output_img = self.lin_model(inter_img)

        if only_output:
            return output_img
        else:
            return output_img, inter_img

    def get_shape(self, lyrs, input_img_shape):
        with torch.no_grad():
            embedder = nn.Sequential(*lyrs)
            output = embedder(torch.zeros(*input_img_shape))
            output_img_shape = output.shape[1:]
            return output_img_shape

def gen_topk_random(sbj_tsnr, k, dim, largest):
    random_idx = torch.rand_like(sbj_tsnr).argsort(dim=dim)
    sbj_tsnr_rndm = torch.take_along_dim(sbj_tsnr, random_idx, dim=dim)
    _, k_random_idx = torch.topk(sbj_tsnr_rndm, k=k, largest=largest, dim=dim)
    k_idx = torch.take_along_dim(random_idx, k_random_idx, dim=dim)
    return k_idx