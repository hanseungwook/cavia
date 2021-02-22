
#################################

from task.Siren_dataio import get_mgrid


class Implicit2DWrapper(Dataset):
    def __init__(self, dataset, sidelength): #, img2fnc=None):

#         if isinstance(sidelength, int):
#             sidelength = (sidelength, sidelength)
#         self.sidelength = sidelength

        self.transform = Compose([ Resize(sidelength), ToTensor(), Normalize(torch.Tensor([0.5]), torch.Tensor([0.5])) ])
        self.dataset = dataset
        self.mgrid = get_mgrid(sidelength)
#         self.img2fnc = img2fnc

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.transform(self.dataset[idx])
        img = img.permute(1, 2, 0).view(-1, self.dataset.img_channels)

        in_dict = {'idx': idx, 'coords': self.mgrid}
        gt_dict = {'img': self.img2fnc(img)}

        return in_dict, gt_dict
    
def img2fnc(mgrid, img):
    subsamples = int(self.test_sparsity)
    rand_idcs = np.random.choice(img.shape[0], size=subsamples, replace=False)
    img_sparse = img[rand_idcs, :]
    coords_sub = self.mgrid[rand_idcs, :]
    in_dict = {'idx': idx, 'coords': self.mgrid, 'img_sub': img_sparse, 'coords_sub': coords_sub}

#     def get_item_small(self, idx):   ### used in dataio.ImageGeneralizationWrapper()
#         img = self.transform(self.dataset[idx])
#         spatial_img = img.clone()
#         img = img.permute(1, 2, 0).view(-1, self.dataset.img_channels)

#         gt_dict = {'img': img}

#         return spatial_img, img, gt_dict
    
    
################


# class CelebA(Dataset):
#     def __init__(self, split, downsampled=False):
#         # SIZE (178 x 218)
#         super().__init__()
#         assert split in ['train', 'test', 'val'], "Unknown split"

#         self.root = '/media/data3/awb/CelebA/kaggle/img_align_celeba/img_align_celeba'
#         csv_path = '/media/data3/awb/CelebA/kaggle/list_eval_partition.csv'

#         self.img_channels = 3
#         self.fnames = []

#         with open(csv_path, newline='') as csvfile:
#             rowreader = csv.reader(csvfile, delimiter=',', quotechar='|')
#             for row in rowreader:
#                 if split == 'train' and row[1] == '0':
#                     self.fnames.append(row[0])
#                 elif split == 'val' and row[1] == '1':
#                     self.fnames.append(row[0])
#                 elif split == 'test' and row[1] == '2':
#                     self.fnames.append(row[0])

#         self.downsampled = downsampled

#     def __len__(self):
#         return len(self.fnames)

#     def __getitem__(self, idx):
#         path = os.path.join(self.root, self.fnames[idx])
#         img = Image.open(path)
#         if self.downsampled:
#             width, height = img.size  # Get dimensions

#             s = min(width, height)
#             left = (width - s) / 2
#             top = (height - s) / 2
#             right = (width + s) / 2
#             bottom = (height + s) / 2
#             img = img.crop((left, top, right, bottom))
#             img = img.resize((32, 32))

#         return img
