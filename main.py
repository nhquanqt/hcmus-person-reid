import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torchvision import transforms
import json
import time

from reid.scores import *
from reid.losses import *
from reid.model import Model
from reid.datasets import Market1501TrainVal, Market1501Test

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

def valid(model, dataset):

    # set model to evaluation mode
    model.eval()
    
    # query features
    query_feat = []
    dataset.query_done = False
    dataset.query_ptr = 0

    while not dataset.query_done:
        images_t, labels_t = dataset.next_batch_query()
        images_t = images_t.to(DEVICE)
        if model.identity_classes is not None:
            feat, _, _ = model(images_t)
        else:
            feat, _ = model(images_t)
        query_feat.append(feat.cpu().detach())

    query_feat = torch.cat(query_feat)
    
    # gallery features
    gallery_feat = []
    dataset.gallery_done = False
    dataset.gallery_ptr = 0

    while not dataset.gallery_done:
        images_t, labels_t = dataset.next_batch_gallery()
        images_t = images_t.to(DEVICE)
        if model.identity_classes is not None:
            feat, _, _ = model(images_t)
        else:
            feat, _ = model(images_t)
        gallery_feat.append(feat.cpu().detach())

    gallery_feat = torch.cat(gallery_feat)
    
    # calculate distance
    dist_mat = euclidean_dist(query_feat, gallery_feat)
    
    # calculate rank-k precision
    cmc_score = cmc(dist_mat, dataset.query_ids, dataset.gallery_ids, dataset.query_cams, dataset.gallery_cams, topk=5, \
                separate_camera_set=False,single_gallery_shot=False,first_match_break=True)
    
    # calculate mean average precision
    mAP = mean_ap(dist_mat, dataset.query_ids, dataset.gallery_ids, dataset.query_cams, dataset.gallery_cams)
    
    # set the model back to train mode
    model.train()
    return mAP, cmc_score

def test(model, dataset, is_global_feature=True, re_ranking=False):
    # Remember to set model to evaluation
    model.eval()

    # query features
    query_feat = []
    dataset.query_done = False
    dataset.query_ptr = 0

    n_images = len(dataset.query_images)
    print(f'number of query images: {n_images}')

    while not dataset.query_done:
        images_t, labels_t = dataset.next_batch_query()
        images_t = images_t.to(DEVICE)
        if model.identity_classes is not None:
            if is_global_feature:
                feat, _, _ = model(images_t)
            else:
                _, feat, _ = model(images_t)
        else:
            if is_global_feature:
                feat, _ = model(images_t)
            else:
                _, feat = model(images_t)

        query_feat.append(feat.cpu().detach())

    query_feat = torch.cat(query_feat)

    print('done')

    # gallery features
    gallery_feat = []
    dataset.gallery_done = False
    dataset.gallery_ptr = 0

    n_images = len(dataset.gallery_images)
    print(f'number of gallery images: {n_images}')

    while not dataset.gallery_done:
        images_t, labels_t = dataset.next_batch_gallery()
        images_t = images_t.to(DEVICE)
        if model.identity_classes is not None:
            if is_global_feature:
                feat, _, _ = model(images_t)
            else:
                _, feat, _ = model(images_t)
        else:
            if is_global_feature:
                feat, _ = model(images_t)
            else:
                _, feat = model(images_t)

        gallery_feat.append(feat.cpu().detach())

    gallery_feat = torch.cat(gallery_feat)

    print('done')

    # calculate distance
    if is_global_feature:
        dist_mat = euclidean_dist(query_feat, gallery_feat)
    else:
        dist_mat = []

        for i in range(0, len(query_feat), 128):
            print(i)
            dist_mat.append(local_dist(query_feat[i:i+128], gallery_feat))

        dist_mat = torch.cat(dist_mat)

    if is_global_feature and re_ranking:
        dist_q_q = euclidean_dist(query_feat, query_feat)
        dist_g_g = euclidean_dist(gallery_feat, gallery_feat)
        dist_mat = re_ranking(dist_mat, dist_q_q, dist_g_g)

    # calculate rank-k precision
    cmc_score = cmc(dist_mat, dataset.query_ids, dataset.gallery_ids, dataset.query_cams, dataset.gallery_cams, topk=5, \
                separate_camera_set=False,single_gallery_shot=False,first_match_break=True)
    
    # calculate mean average precision
    mAP = mean_ap(dist_mat, dataset.query_ids, dataset.gallery_ids, dataset.query_cams, dataset.gallery_cams)

    return cmc_score, mAP

def main():
    # load configuration
    cfg = json.load(open('config.json'))

    print(cfg)

    assert cfg['test_by_global_feature'] or not cfg['re_ranking'], 'The program could perform re-ranking only with global feature.'

    root = cfg['root']

    n_epochs = cfg['n_epochs']

    lr = cfg['lr']
    weight_decay = cfg['weight_decay']

    g_loss_weight = cfg['g_loss_weight']
    g_margin = cfg['g_margin']

    l_loss_weight = cfg['l_loss_weight']
    l_margin = cfg['l_margin']

    id_loss_weight = cfg['id_loss_weight']
    id_classes = cfg['id_classes']

    batch_size = cfg['batch_size']

    # load dataset
    transform_trainval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([256, 128]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    market1501_trainval = Market1501TrainVal(root, transform_trainval, batch_size=batch_size)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([256, 128]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    market1501_test = Market1501Test(root, transform_test)

    # load model
    if id_loss_weight > 0:
        model = Model(last_conv_stride=1, identity_classes=id_classes).to(DEVICE)
    else:
        model = Model(last_conv_stride=1).to(DEVICE)

    if cfg['is_test_only']:
        model.load_state_dict(torch.load(cfg['model_weight_path']))

        cmc_score, mAP_score = test(model, market1501_test, cfg['test_by_global_feature'])
        print('-----------------------TEST MODEL------------------------')
        print(f'Rank-1 score: {cmc_score[0]}')
        print(f'Rank-5 score: {cmc_score[4]}')
        print(f'mAP score: {mAP_score}')
        return

    # load losses, optimizer and scheduler
    cur_epoch = 0

    g_tri_loss = TripletLoss(g_margin)
    l_tri_loss = TripletLoss(l_margin)
    if id_loss_weight > 0:
        id_criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if cfg['is_resume_training']:
        checkpoint = torch.load(cfg['checkpoint_path'])

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        cur_epoch = checkpoint['epoch']

    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

    print(f'number of person-ids for training: {len(market1501_trainval.person_id_list)}')
    print(f'number of training images: {len(market1501_trainval.train_images)}')
    print(f'number of query images for validation: {len(market1501_trainval.query_images)}')
    print(f'number of gallery images for validaion: {len(market1501_trainval.gallery_images)}')

    # start training
    model.train()

    for epoch in range(cur_epoch, n_epochs):
        print(f'epoch: {epoch}')
        market1501_trainval.start_over()

        total_loss = []
        total_g_loss = []
        total_g_prec = []
        total_g_sm = []
        total_g_dist_ap = []
        total_g_dist_an = []

        total_l_loss = []
        total_l_prec = []
        total_l_sm = []
        total_l_dist_ap = []
        total_l_dist_an = []

        total_id_loss = []

        iter = 0
        t_now = time.time()
        while not market1501_trainval.epoch_done:
            images_t, labels_t = market1501_trainval.next_batch()
            images_t = Variable(images_t.to(DEVICE))

            if id_loss_weight > 0:
                g_feat, l_feat, logits = model(images_t)
            else:
                g_feat, l_feat = model(images_t)

            g_loss, g_dist_ap, g_dist_an = global_loss(g_tri_loss, g_feat, labels_t)
            l_loss, l_dist_ap, l_dist_an = local_loss(l_tri_loss, l_feat, labels_t)

            if id_loss_weight > 0:
                id_loss = id_criterion(logits, labels_t.long().to(DEVICE))
                loss = g_loss_weight * g_loss + l_loss_weight * l_loss + id_loss_weight * id_loss
            else:
                loss = g_loss_weight * g_loss + l_loss_weight * l_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log
            total_loss.append(loss.item())

            total_g_loss.append(g_loss.item())
            total_g_prec.append((g_dist_an > g_dist_ap).data.float().mean().item())
            total_g_sm.append((g_dist_an > g_dist_ap + g_margin).data.float().mean().item())
            total_g_dist_ap.append(g_dist_ap.data.float().mean().item())
            total_g_dist_an.append(g_dist_an.data.float().mean().item())

            total_l_loss.append(l_loss.item())
            total_l_prec.append((l_dist_an > l_dist_ap).data.float().mean().item())
            total_l_sm.append((l_dist_an > l_dist_ap + l_margin).data.float().mean().item())
            total_l_dist_ap.append(l_dist_ap.data.float().mean().item())
            total_l_dist_an.append(l_dist_an.data.float().mean().item())

            if id_loss_weight > 0:
                total_id_loss.append(id_loss.item())

            iter += 1
        print(f'ran in {(time.time() - t_now)}s')
        if id_loss_weight > 0:
            print(f'mean_loss = {np.mean(total_loss)}, mean_g_loss = {np.mean(total_g_loss)}, mean_l_loss = {np.mean(total_l_loss)}, mean_id_loss = {np.mean(total_id_loss)}')
        else:
            print(f'mean_loss = {np.mean(total_loss)}, mean_g_loss = {np.mean(total_g_loss)}, mean_l_loss = {np.mean(total_l_loss)}')
        print(f'mean_g_prec = {np.mean(total_g_prec)}, mean_l_prec = {np.mean(total_l_prec)}')
        print(f'mean_g_sm = {np.mean(total_g_sm)}, mean_l_sm = {np.mean(total_l_sm)}')
        print(f'mean_g_dist_ap = {np.mean(total_g_dist_ap)}, mean_l_dist_ap = {np.mean(total_l_dist_ap)}')
        print(f'mean_g_dist_an = {np.mean(total_g_dist_an)}, mean_l_dist_an = {np.mean(total_l_dist_an)}')
        print(f'learning_rate = {scheduler.get_last_lr()}')
        
        scheduler.step()
        
        if (epoch + 1) % cfg['epochs_per_valid'] == 0:
            mAP, cmc_score = valid(model, market1501_trainval)
            print(f'--------------------------')
            print(f'validation: mAP={mAP}, rank1={cmc_score[0]}, rank5={cmc_score[4]}')
            print(f'--------------------------')
        
        if (epoch + 1) % cfg['epochs_per_checkpoint'] == 0:
            # save model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'checkpoint_s1_{epoch+1}.pth')

    # test model
    cmc_score, mAP_score = test(model, market1501_test, cfg['test_by_global_feature'])
    print('-----------------------TEST MODEL------------------------')
    print(f'Rank-1 score: {cmc_score[0]}')
    print(f'Rank-5 score: {cmc_score[4]}')
    print(f'mAP score: {mAP_score}')


if __name__ == '__main__':
    main()