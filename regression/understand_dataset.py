
# lv3: dataset f( .  , rnd, type)  -> dataset   (e.g. MNIST vs fMNIST)
# lv2: label   f(dataset, rnd, type)  -> label              -> Loss( theta | dataset )           -> averaged over labels
# lv1: image   f(label, rnd, type) -> image                 -> Loss( ctx1 | label | theta)       -> averaged over image
# lv0: pixel   f(image, rnd, type) -> xy, pixel (target)    -> Loss( ctx0 | image | ctx1, theta) -> averaged over xy E_xy[L(xy)]


# LQR
# lv2: dyanamics   f(.,   rnd, type)   -> A,B               -> Loss( . | . ) -> averaged over A,B
# lv1: target      f(., rnd, type)     -> target            -> Loss( ctx1 | A,B | . ) -> averaged over target
# lv0: x0          f(., rnd, type)     -> x0                -> Loss( ctx0 | target | ctx1, A,B) -> averaged over x0

###############################################

# for i, task_batch in enumerate(dataloader):
#     for _ in range(for_iter):
#         loss = model(task_batch, level, status=status)[0]     # Loss to be optimized

#         if cur_iter >= max_iter:    # train/optimize up to max_iter # of batches
            
            
# k_batch : total batch (k-shot)
# n_batch : mini batch  ()

# # max_iter: total iteration for the level's inner update. so it does not use the whole batch..  (1 epoch)
# # for_iter should be 1 ??

###############################################

# type = train/test/valid

# elif task == 'mnist':
# if len(classes) <= 0:
#     classes = list(range(0, 10))
# task_func_dict['train'].extend([partial(sample_mnist_img_fnc, l) for l in classes])



# def get_samples(task, total_batch, sample_type):
# #     sample_type = 'train' or 'test'  
#     # only useful for task_dict .. only level2??
    
#     if isinstance(task, dict):
#         print('level2 task_dict', task)
#         task_list = task[sample_type]
#         if task_list == []: # if sample_type == 'test' and task_ is empty 
#             task_list = task['train']  # then use train task
        
#         if isinstance(task_list, list):
#             assert total_batch <= len(task_list)
#             task_samples = random.sample(task_list, total_batch)  # subsampling from the list of tasks
#         else:
#             error()
        
#     # Levels below 2
#     else:
#         print('level1 task function', task)
# #         sample_type ('train' vs 'test') is not needed for level 1? 
#         task_samples = [task(None) for _ in range(total_batch)]
# #         task_samples = [task(sample_type) for _ in range(total_batch)]
        
# #     print('sample_type', sample_type)  # Delete this
# #     print('task_samples', task_samples)
#     return task_samples


def sample_mnist_img_fnc(label, sample_type):
    img = get_mnist_img(sample_type, label)
    t_fn = partial(img_target_function, img)

    return img_input_function, t_fn



def img_target_function(img, coordinates):
    c = copy.deepcopy(coordinates)
    
    # Denormalize coordinates
    c[:, 0] *= img_size[0]
    c[:, 1] *= img_size[1]

    # Usual H x W x C img dimensions
    if img.shape[2] == 3 or img.shape[2] == 1:
        pixel_values = img[c[:, 0].long(), c[:, 1].long(), :]    
    # Pytorch C x H x W img dimensions
    elif img.shape[0] == 3 or img.shape[0] == 1:
        pixel_values = img[:, c[:, 0].long(), c[:, 1].long()].permute(1, 0) 

    return pixel_values



def img_input_function(batch_size, order_pixels=False):
    if order_pixels:
        flattened_indices = list(range(img_size[0] * img_size[1]))[:batch_size]
    else:
        # Returning full range (in sorted order) if batch size is the full image size
        if batch_size == 0:
            flattened_indices = list(range(img_size[0] * img_size[1]))
        else: 
            flattened_indices = np.random.choice(list(range(img_size[0] * img_size[1])), batch_size, replace=False)

    x, y = np.unravel_index(flattened_indices, (img_size[0], img_size[1]))
    coordinates = np.vstack((x, y)).T
    coordinates = torch.from_numpy(coordinates).float()
    
    # Normalize coordinates
    coordinates[:, 0] /= img_size[0]
    coordinates[:, 1] /= img_size[1]
    return coordinates



def get_mnist_img(sample_type, label):
    global task
    imgs = None

    if not (train_imgs and test_imgs) or task != 'mnist':
        load_mnist_imgs()
        task = 'mnist'
    
    # Read from global variables
    if sample_type == 'train':
        imgs = train_imgs
    elif sample_type == 'test':
        imgs = test_imgs
    else:
        raise Exception('Wrong sampling type')

    # Get indices of given label in the dataset
    labels = imgs.targets.numpy()
    img_idx = np.random.choice(np.where(labels == label)[0], size=1)[0]

    img, _ = imgs[img_idx]
    img = img.permute(1, 2, 0)

    return img

