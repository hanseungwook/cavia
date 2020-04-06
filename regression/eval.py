def eval_cavia(args, model, task_family, num_updates, n_tasks=100, task_type='train', return_gradnorm=False):
    # get the task family
    input_range = task_family.get_input_range().to(args.device)

    # logging
    losses = []
    gradnorms = []

    # --- inner loop ---
    for t in range(n_tasks):

        # sample a task
        target_fnc = task_family.sample_task()

        context = model.reset_context()
        # ------------ update context on current task ------------
        for _ in range(1, num_updates + 1):
            context = inner_update(args, model, task_family, target_fnc)

        # compute true loss on entire input range
        model.eval()
        losses.append(eval_model(model, context, input_range, target_fnc).detach().item())
        model.train()

    losses_mean = np.mean(losses)
    losses_conf = st.t.interval(0.95, len(losses) - 1, loc=losses_mean, scale=st.sem(losses))
    if not return_gradnorm:
        return losses_mean, np.mean(np.abs(losses_conf - losses_mean))
    else:
        return losses_mean, np.mean(np.abs(losses_conf - losses_mean)), np.mean(gradnorms)

def eval_1hot(args, model, task_family, num_updates, n_tasks=25, task_type='train', return_gradnorm=False):    
    # logging
    losses = []
    gradnorms = []

    # Optimizer for encoder (context)
    inner_optimizer = optim.SGD(model.encoder.parameters(), args.lr_inner)

    # Sample tasks
    target_functions, labels_1hot = task_family.sample_tasks_1hot(total_num_tasks=n_tasks*4, batch_num_tasks=n_tasks, batch_size=args.k_shot_eval)

    # Only if task is valid or test, reinitialize model and perform inner updates
    if task_type != 'train':
        print('Reinit model {}'.format(task_type))
        # Reinitialize one hot encoder for evaluation mode
        model.encoder.reinit(args.num_context_params, task_family.total_num_tasks)
        model.encoder.train()
        
        # Inner loop steps
        for _ in range(1, num_updates + 1):
            inputs = task_family.sample_inputs(args.k_meta_test * n_tasks, args.use_ordered_pixels).to(args.device)
            targets = create_targets(inputs, target_functions, n_tasks)
            labels = create_1hot_labels(labels_1hot, n_tasks)

            loss = update_step_1hot(model, inner_optimizer, inputs, targets, labels, n_tasks)
            # print('{} loss: {}'.format(task_type, loss))

     # Get entire range's input and respective 1 hot labels
    input_range = task_family.get_input_range().to(args.device).repeat(n_tasks, 1)
    bs = int(input_range.shape[0] / n_tasks)
    input_range_1hot = torch.cat([task_family.create_input_range_1hot_labels(batch_size=bs, cur_label=labels_1hot[t, 0, :]) for t in range(n_tasks)])
    input_range_targets = create_targets(input_range, target_functions, n_tasks)
    
    # compute true loss on entire input range
    model.eval()
    losses.append(eval_model_1hot(model, input_range, input_range_targets, input_range_1hot).item())
    #viz_pred(model, input_range[:bs], input_range_targets[:bs], input_range_1hot[:bs], task_type)
    model.train()

    losses_mean = np.mean(losses)
    losses_conf = st.t.interval(0.95, len(losses) - 1, loc=losses_mean, scale=st.sem(losses))
    if not return_gradnorm:
        return losses_mean, np.mean(np.abs(losses_conf - losses_mean))
    else:
        return losses_mean, np.mean(np.abs(losses_conf - losses_mean)), np.mean(gradnorms)