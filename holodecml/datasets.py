from torch.utils.data import Dataset
from holodecml.transforms import LoadTransformations
from holodecml.propagation import WavePropagator
from holodecml.propagation import UpsamplingPropagator
import yaml
import torch
import numpy as np
import random


class LoadHolograms(Dataset):
    def __init__(
        self,
        file_path,
        n_bins=1000,
        shuffle=False,
        device="cpu",
        transform=False,
        lookahead=2,  # this is really the number of planes
        step_size=32,
        tile_size=32,
        balance=True,
        output_lst=None,  # output functions to apply to complex field (abs, real, imag)
        deweight=1e-3,  # deweighting applied to nonparticle pixels in weight mask
    ):

        # num of waveprop windows
        self.n_bins = n_bins
        # device used
        self.device = device
        # shuffle frames
        self.shuffle = shuffle
        # number of planes to reconstruct around the z position
        # for this class lookahead must be 2 or more
        assert lookahead > 1
        self.lookahead = lookahead
        self.z_bck_idx = int(np.floor((self.lookahead-1)/2)) # planes back from indexed plane
        self.z_fwd_idx = int(np.ceil((self.lookahead-1)/2))+1  # planes in front of indexed plane
        # wavepropagator object on device
        self.propagator = WavePropagator(
            file_path,
            n_bins=n_bins,
            device=device,
            step_size=step_size,
            tile_size=tile_size,
        )
        self.transform = transform
        self.indices = [
            (x, y)
            for x in self.propagator.h_ds.hologram_number
            for y in range(self.n_bins - self.lookahead)
        ]

        self.tile_size = tile_size

        if output_lst is None:
            self.output_lst = [torch.abs, torch.angle]
        else:
            self.output_lst = output_lst

        self.deweight = deweight

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        if self.shuffle:
            idx = random.choice(range(self.__len__()))

        h_idx, z_idx = self.indices[idx]
        #z_props = self.propagator.z_centers[z_idx : z_idx + self.lookahead + 1]
        #plane_indices = np.arange(z_idx, z_idx + self.lookahead + 1)

        # select hologram
        image = (
            self.propagator.h_ds["image"]
            .isel(hologram_number=h_idx)
            .values.astype(float)
        )

        # propagate
        z_slc = slice(z_idx-self.z_bck_idx,
                      z_idx+self.z_fwd_idx)
        image = self.propagator.torch_holo_set(
            torch.from_numpy(image).to(self.device),
            torch.FloatTensor([self.propagator.z_centers[z_slc,np.newaxis,np.newaxis]*1e-6]).to(self.device)
        ) # image now has dims (z-planes, x, y)

        ch_lst = []
        for fnc in self.output_lst:
            ch_lst.append(fnc(image))
        in_channels = torch.cat(ch_lst,dim=1)  # stack the channels in the same axis as the z planes

        # image = torch.abs(image).cpu().numpy() ### the transforms all need to be done with torch and not numpy in a future version.
        # image, hflip, vflip = self.apply_transforms(image)
        num_particles, part_mask, depth_mask, weight_mask = self.create_mask(h_idx, z_idx)
        # mask = torch.flip(mask, [0]) if hflip else mask
        # mask = torch.flip(mask, [1]) if vflip else mask
        return self.pad_images_and_mask(in_channels, part_mask, depth_mask, weight_mask)

    
    def create_mask(self, h_idx, z_idx):

        hid = h_idx + 1
        hid_mask = self.propagator.h_ds["hid"] == hid

        # Filter particles based on h_idx
        x_part = self.propagator.h_ds["x"].values[hid_mask]
        y_part = self.propagator.h_ds["y"].values[hid_mask]
        z_part = self.propagator.h_ds["z"].values[hid_mask]
        d_part = self.propagator.h_ds["d"].values[hid_mask]
        
        # z_indices = np.digitize(z_part, self.propagator.z_bins) - 1
        # Initialize the UNET mask
        unet_mask = np.zeros((self.propagator.x_arr.shape[0], self.propagator.y_arr.shape[0]))
        depth_mask = np.zeros((self.propagator.x_arr.shape[0], self.propagator.y_arr.shape[0]))
        weight_mask = np.zeros((self.propagator.x_arr.shape[0], self.propagator.y_arr.shape[0]))+self.deweight
        
        num_particles = 0 

        # find particles that are contained between the first and last planes
        cond = np.where((z_part >= self.propagator.z_centers[z_idx-self.z_bck_idx]) & \
                        (z_part <= self.propagator.z_centers[z_idx+self.z_fwd_idx]))

        # if z_idx in z_indices:
        # cond = np.where(z_idx == z_indices)
        if np.size(cond[0]) > 0:
            x_part = x_part[cond]
            y_part = y_part[cond]
            z_part = z_part[cond]
            d_part = d_part[cond]

            # Build the UNET mask using vectorized operations
            for part_idx in range(len(cond[0])):
                z_diff = z_part[part_idx] - self.propagator.z_centers[z_idx] # z distance from reference plane
                y_diff = (self.propagator.y_arr[None, :] * 1e6 - y_part[part_idx])
                x_diff = (self.propagator.x_arr[:, None] * 1e6 - x_part[part_idx])
                d_squared = (d_part[part_idx] / 2)**2
                part_pxl_idx = np.where((y_diff**2 + x_diff**2) < d_squared)
                unet_mask[part_pxl_idx] = 1.0
                # unet_mask += ((y_diff**2 + x_diff**2) < d_squared).astype(float)
                depth_mask[part_pxl_idx] = z_diff
                weight_pxl_idx = np.where((y_diff**2 + x_diff**2) < 4*d_squared)
                weight_mask[weight_pxl_idx] = 1.0
                num_particles += 1
                
            xp = np.digitize(x_part[part_idx], 1e6 * self.propagator.x_arr, right=True)
            yp = np.digitize(y_part[part_idx], 1e6 * self.propagator.y_arr, right=True)
            #print(xp, yp)

        return num_particles, torch.from_numpy(unet_mask), torch.from_numpy(depth_mask), torch.from_numpy(weight_mask)
    
    
    def get_particle(self, h_idx):
        indices = np.where(self.propagator.h_ds["hid"] == h_idx + 1)
        d_locations = self.propagator.h_ds["d"].values[indices]
        x_locations = self.propagator.h_ds["x"].values[indices]
        y_locations = self.propagator.h_ds["y"].values[indices]
        z_locations = self.propagator.h_ds["z"].values[indices]
    
        xp = np.digitize(x_locations, 1e6 * self.propagator.x_arr, right=True)
        yp = np.digitize(y_locations, 1e6 * self.propagator.y_arr, right=True)
        zp = z_locations
        dp = d_locations

        for (x, y, z, d) in zip(xp, yp, zp, dp):
            yield x, y, z, d
            
    def apply_transforms(self, image):
        im = {
            "image": np.expand_dims(image, 0),
            "horizontal_flip": False,
            "vertical_flip": False,
        }
        # add transformations here
        if self.transform:
            for image_transform in self.transform:
                im = image_transform(im)
        return im["image"], im["horizontal_flip"], im["vertical_flip"]
        
    
    def pad_images_and_mask(self, image_stack, mask1, mask2, mask3, target_height = 4896, target_width = 3264):
        """
        Pad the image_stack and mask with zeros to sizes (num_images, channels, 4896, 3264) and (4896, 3264) respectively using PyTorch.

        Parameters:
            image_stack (torch.Tensor): The input image stack with shape (num_images, channels, height, width).
            mask (torch.Tensor): The input mask with shape (height, width).

        Returns:
            tuple: A tuple containing the padded image_stack and mask as torch.Tensors.
        """
        current_height, current_width = image_stack.size(-2), image_stack.size(-1)
        pad_height = max(target_height - current_height, 0)
        pad_width = max(target_width - current_width, 0)

        # Pad the image_stack
        mean_image = 0 #torch.mean(image_stack)
        padded_image_stack = torch.nn.functional.pad(image_stack, (0, pad_width, 0, pad_height), mode='constant', value=mean_image)

        # Pad the mask
        padded_mask1 = torch.nn.functional.pad(mask1, (0, pad_width, 0, pad_height), mode='constant', value=0)
        padded_mask2 = torch.nn.functional.pad(mask2, (0, pad_width, 0, pad_height), mode='constant', value=0)
        padded_mask3 = torch.nn.functional.pad(mask3, (0, pad_width, 0, pad_height), mode='constant', value=0)

        return padded_image_stack, padded_mask1, padded_mask2, padded_mask3
    
    
class UpsamplingReader(Dataset):

    def __init__(self, conf=None, data_path=None, transform=None, max_size=10000, device="cpu"):

        config = conf["data"]
        n_bins = config["n_bins"]
        #data_path = config["data_path"]
        tile_size = config["tile_size"]  # size of tiled images in pixels
        # amount that we shift the tile to make a new tile
        step_size = config["step_size"]
        # UNET gaussian marker width (standard deviation) in um
        marker_size = config["marker_size"]
        transform_mode = "None" if "transform_mode" not in config else config["transform_mode"]

        self.part_per_holo = config["total_positive"]
        self.empt_per_holo = config["total_negative"]
        self.color_dim = conf["model"]["in_channels"]

        self.prop = UpsamplingPropagator(
            data_path,
            n_bins=n_bins,
            tile_size=tile_size,
            step_size=step_size,
            marker_size=marker_size,
            transform_mode=transform_mode,
            device=device
        )

        self.xy = []
        self.max_size = max_size
        self.transform = transform

    def __getitem__(self, h_idx):

        h_idx = int(self.prop.h_ds["hid"].values[h_idx]) - 1

        if len(self.xy) > 0:
            random.shuffle(self.xy)
            x, y = self.xy.pop()
            return x, y

        data = self.prop.get_reconstructed_sub_images(
            h_idx, self.part_per_holo, self.empt_per_holo
        )
        for idx in range(len(data[0])):
            # result_dict["label"].append(int(data[0][idx]))
            image = np.expand_dims(np.abs(data[1][idx]), 0)
            if self.color_dim == 2:
                phase = np.expand_dims(np.angle(data[1][idx]), 0)
                image = np.vstack([image, phase])
            mask = data[4][idx]
            image, mask = self.apply_transforms(image, mask)
            self.xy.append((image, mask))

        random.shuffle(self.xy)
        x, y = self.xy.pop()
        return x, y

    def apply_transforms(self, image, mask):

        if self.transform == None:

            image = torch.tensor(image, dtype=torch.float)
            mask = torch.tensor(mask, dtype=torch.int)

            return image, mask

        im = {
            "image": image,
            "horizontal_flip": False,
            "vertical_flip": False
        }

        for image_transform in self.transform:
            im = image_transform(im)

        # Update the mask if we flipped the original image
        if im["horizontal_flip"]:
            mask = np.flip(mask, axis=0)
        if im["vertical_flip"]:
            mask = np.flip(mask, axis=1)

        image = torch.tensor(im["image"], dtype=torch.float)
        mask = torch.tensor(mask.copy(), dtype=torch.int)

        return image, mask

    def __len__(self):
        return len(list(self.prop.h_ds["hid"].values))

    
def unpad_images_and_mask(padded_image_stack, padded_mask, original_height=4872, original_width=3248):
    """
    Unpad the padded image_stack and mask to their original sizes.

    Parameters:
        padded_image_stack (torch.Tensor): The padded image stack with shape (num_images, channels, padded_height, padded_width).
        padded_mask (torch.Tensor): The padded mask with shape (padded_height, padded_width).
        original_height (int): The original height of the image stack and mask.
        original_width (int): The original width of the image stack and mask.

    Returns:
        tuple: A tuple containing the unpadded image_stack and mask as torch.Tensors.
    """

    # Unpad the image_stack
    unpadded_image_stack = padded_image_stack[:, :, :original_height, :original_width].clone()

    # Unpad the mask
    unpadded_mask = padded_mask[:, :original_height, :original_width].clone()

    return unpadded_image_stack, unpadded_mask
    
    
if __name__ == "__main__":
    with open("/glade/work/schreck/repos/HOLO/dev/holodec-ml/results/manopt/model.yml") as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    lookahead = 0 #int(conf["data"]["lookahead"])
    conf["model"]["in_channels"] = 2 * (lookahead + 1)

    n_bins = int(conf["data"]["n_bins"])
    tile_size = int(conf["data"]["tile_size"])
    step_size = int(conf["data"]["step_size"])

    train_dataset = LoadHolograms(
        "/glade/p/cisl/aiml/ai4ess_hackathon/holodec/synthetic_holograms_500particle_gamma_4872x3248_training.nc", 
        shuffle = False, 
        device = "cpu", 
        n_bins = n_bins, 
        transform = LoadTransformations(conf["transforms"]["training"]), 
        lookahead = lookahead, 
        tile_size = tile_size, 
        step_size = step_size
    )
    random_integer = random.randint(0, train_dataset.__len__())
    print("Id", random_integer)
    x, y = train_dataset.__getitem__(random_integer)
    print(x.shape, y.shape)
    print("Mask sum", y.sum())