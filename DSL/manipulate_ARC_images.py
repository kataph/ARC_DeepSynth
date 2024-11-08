from PIL import Image
import requests
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgb
from tqdm import tqdm
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from math import floor
from PIL import Image, ImageOps
from scipy import ndimage, datasets
import numpy as np

# from transformers import AutoProcessor, CLIPVisionModel

# model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

inp = [[1,0,0,0],[1,1,0,0],[1,0,0,0],[0,0,0,0]]
out = [[0,1,0,0],[0,1,1,0],[0,1,0,0]]

def to_tuple(gr: list[list[int]]):
    return tuple(tuple(row) for row in gr)

inp = to_tuple(inp)
out = to_tuple(out)

# last one is padding color
colors = ['black', 'gold', 'red', 'green', 'blue', 'pink', 'orange', 'violet', 'yellow', 'cyan', 'white']
# RGB_colors = [to_rgb(c) for c in colors] # useless...
ARCcmap = ListedColormap(colors)

def couple_and_pad(inp,out, pad = 10):
    ix=len(inp)
    iy=len(inp[0])
    ox=len(out)
    oy=len(out[0])
    mx=max(ix,ox)
    my=max(iy,oy)
    coupled = []
    for i in range(mx):
        coupled.append((inp[i] if i < ix else tuple([pad for z in range(iy)])) + (pad,)*(max(mx//5,1)) + (out[i] if i < ox else tuple([pad for z in range(oy)])))
    return to_tuple(coupled)

def couple_resize_pad(inp, out, pad = 10):
    cc = couple_and_pad(inp,out)
    ccx=len(cc)
    ccy=len(cc[0])
    ccm=max(ccx,ccy)
    factor=224/ccm
    def zoom(a, factor):
        a = np.asarray(a)
        sx, sy = (factor * dim for dim in a.shape)
        X, Y = np.ogrid[0:sx, 0:sy]
        return a[X//factor, Y//factor]
    #ccc=ndimage.zoom(cc, factor, order=0)
    ccc=zoom(cc, 224//ccm)
    cccx=len(ccc)
    try:
        cccy=len(ccc[0])
    except:
        print(inp, out, pad)
        print(cc)
        print(ccx)
        print(ccy)
        print(cccx)
        print(ccc)
        print(224//ccm)
        print(locals())
    cccm=max(cccx,cccy)
    # assert cccm==224
    if cccx==224 and cccy < 224:
        q=(224-cccy)//2
        r=(224-cccy)%2
        ppad=((0,0),(q,q+r))
    elif cccy==224 and cccx < 224:
        q=(224-cccx)//2
        r=(224-cccx)%2
        ppad=((q,q+r),(0,0))
    elif cccy < 224 and cccx < 224:
        q=(224-cccx)//2
        r=(224-cccx)%2
        q2=(224-cccy)//2
        r2=(224-cccy)%2
        ppad=((q,q+r),(q2,r2))
        # ppad=((q2,r2),(q,q+r))
    else:
        ppad=((0,0),(0,0))
    pcc=np.pad(ccc, ppad, constant_values=pad)
    # plt.matshow(cc,cmap=ARCcmap)
    # plt.matshow(ccc,cmap=ARCcmap)
    # plt.matshow(pcc,cmap=ARCcmap)
    # plt.show()
    return pcc

def padding(img, desired_size, fill=0):
    delta_width = desired_size[0] - img.size[0]
    delta_height = desired_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding, fill=fill)


def resize_with_padding(img, expected_size, fill = 0):
    img.thumbnail((expected_size[0], expected_size[1]))
    # print(img.size)
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding, fill = fill)

def from_2Dgrid_to_pil_image(grid, cmap):
    return Image.fromarray(np.uint8(cmap(grid)*255)) # DO NOT PUT RGB MODE! COLORMAP ENTAILS RGBA

def from_RGB_tensor_to_pil_image(rgb_tensor):
    """takes C.H.W as input"""
    return to_pil_image(rgb_tensor)

def upscale_with_PIL(image: Image, target = 224):
    small_size = image.size
    if small_size[0] < small_size[1]:
        max_size=small_size[1]
        big_size = (floor(small_size[0]*(target/small_size[1])), target)
    else:
        max_size=small_size[0]
        big_size = (target, floor(small_size[1]*(target/small_size[0])))
    return image.resize(big_size, Image.NEAREST)


def from_grid_and_cmap_to_image_tensor(grid, cmap): # faster than manual
    """grid is NOT numpy-like"""
    return  pil_to_tensor(from_2Dgrid_to_pil_image(np.asarray(grid), cmap = ARCcmap))

#This is faster!
def from_coupling_to_tensor_with_PIL(inp, out):
    """outputs C.W.H"""
    coupled = couple_and_pad(inp,out)
    img = from_2Dgrid_to_pil_image(coupled, cmap=ARCcmap)
    img = img.convert("RGB")
    img = upscale_with_PIL(img, target = 224)
    img = padding(img, (224,224), fill='white')
    return pil_to_tensor(img)

def from_coupling_to_tensor_without_PIL_rescaling(inp, out):
    resized = couple_resize_pad(inp,out)
    img = from_2Dgrid_to_pil_image(resized, cmap=ARCcmap)
    img = img.convert("RGB")
    return pil_to_tensor(padding(img, (224,224), fill='white'))

#returns expected 
# print(from_coupling_to_tensor_with_PIL(inp,out).shape)
# from_RGB_tensor_to_pil_image(from_coupling_to_tensor_with_PIL(inp,out)).show();s()

# x=from_coupling_to_tensor_with_PIL(inp, out)
# print(x.shape)
# for i in tqdm(range(1000)):
#     x=from_coupling_to_tensor_with_PIL(inp, out)
# for i in tqdm(range(1000)):
#     x=from_coupling_to_tensor_without_PIL_rescaling(inp, out)
# stop()


def visualize_input_output_example_together(grid_in, grid_out):
    fig_tog, axes = plt.subplots(1,2,sharex=False,sharey=False)
    
    axes[0].set_axis_off()  # corresponds to plt.subplot(x).axis('off')
    axes[1].set_axis_off()
    
    axes[0].matshow(grid_in, cmap=ARCcmap, vmin=0, vmax=9)
    axes[1].set_xlim(axes[0].get_xlim())
    axes[1].set_ylim(axes[0].get_ylim())
    axes[1].matshow(grid_out, cmap=ARCcmap, vmin=0, vmax=9)
    
    return fig_tog








if __name__ == "__main__":
    ARCimg = Image.open("temp.png")
    small_size = ARCimg.size
    if small_size[0] < small_size[1]:
        max_size=small_size[1]
        big_size = (floor(small_size[0]*(224/small_size[1])), 224)
    else:
        max_size=small_size[0]
        big_size = (224, floor(small_size[1]*(224/small_size[0])))

    ARCimg = ARCimg.resize(big_size)
    #ARCimg = ARCimg.resize((224,224)) # returns stretched to fill 224.224 square
    ARCimg = padding(ARCimg, (224,224), fill='white') # returns efficient padded to fill 224.224
    ARCimg.save('temp_resized.png')


    inputs = processor(images=ARCimg, return_tensors="pt")

    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    pooled_output = outputs.pooler_output  # pooled CLS states
    print(pooled_output[0][:10])

    for u in tqdm(range(100)): 
        im=couple_resize_pad(inp,out)
        im=from_2Dgrid_to_pil_image(im, cmap=ARCcmap)

    for c in tqdm(range(100)):
        fig_tog = visualize_input_output_example_together(inp, out)
        fig_tog.set_size_inches(w=1,h=1)
        plt.close(fig_tog)

    for c in tqdm(range(100)):
        fig_tog.savefig(f'.//ARC_images//{hash(inp)}_{hash(out)}_{c}.png', 
                    format = 'png', 
                    transparent = False, 
                    bbox_inches= 'tight', 
                    pad_inches=0.0, 
                    facecolor='auto', 
                    edgecolor='auto',
                    dpi=224
                    )   # save the figure to file
    # plt.show()
        plt.close(fig_tog)