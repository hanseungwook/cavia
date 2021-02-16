
# lv3: dataset f( .  , rnd, type)    -> dataset    (e.g. MNIST vs fMNIST)
# lv2: label   f(dataset, rnd, type) -> label                -> Loss( theta | dataset )           -> averaged over labels
# lv1: image   f(label, rnd, type)   -> image                -> Loss( ctx1 | label | theta)       -> averaged over image
# lv0: pixel   f(image, rnd, type)   -> xy, pixel (target)   -> Loss( ctx0 | image | ctx1, theta) -> averaged over xy E_xy[L(xy)]


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

