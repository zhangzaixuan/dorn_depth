# coding=utf-8
"""
@Author: jing
@Email:
"""
import os
import time
import socket
from datetime import datetime
import torch
# 将cudann 加速设置为false,30系显卡适配driver驱动和cuda版本较新
torch.backends.cudnn.enabled = False

# 非确定性算法
# torch.backends.cudnn.enabled =True
# torch.backends.cudnn.benchmark = True
from tensorboardX import SummaryWriter
from DORNnet import DORN
from load_data import getNYUDataset, get_depth_sid
import m_utils
from m_utils import ordLoss, update_ploy_lr, save_checkpoint
from error_metrics import AverageMeter, Result

# 模型初始化学习率
init_lr = 0.0001
# 网络初始化权值
momentum = 0.9
# 模型迭代轮次
epoches = 140
# 模型训练参数个数
batch_size = 16
# batch_size = 2
# 模型迭代次数,一次mini-batch的前向和后向传播
max_iter = 9000000
# max_iter = 90000000

resume = False  # 是否有已经保存的模型,否
model_path = '.\\run\\checkpoint-119.pth.tar'  # 注意修改加载模型的路径
# model_path = '.\\run\\model_best.pth.tar' # 注意修改加载模型的路径
output_dir = '.\\run'


def main():
    """
    程序主函数,模型训练，迭代保存模型文件
    :return:null
    """
    # 数据加载器，这里为图片加载器，train,val,test 分别为训练，验证和测试对应的图片加载器
    # getNYUDataset() 为数据集获取函数
    train_loader, val_loader, test_loader = getNYUDataset()
    print("已经获取数据")
    best_result = Result()
    # 先将结果设置成最差，在每轮迭代中获取最佳参数
    best_result.set_to_worst()
    # resume为true,加载已经训练好的模型，模型首次训练没有已有模型，global var resume 已经设置为false
    if resume:
        # TODO
        # best result应当从保存的模型中读出来
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
        model_dict = checkpoint['model']
        # model = DORN()
        # model.load_state_dict(model_dict)
        model = checkpoint['model']
        # 使用SGD进行优化
        # in paper, aspp module's lr is 20 bigger than the other modules
        aspp_params = list(map(id, model.aspp_module.parameters()))
        base_params = filter(lambda p: id(p) not in aspp_params, model.parameters())
        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        optimizer = torch.optim.SGD([
            {'params': base_params},
            {'params': model.aspp_module.parameters(), 'lr': init_lr * 20},
        ], lr=init_lr, momentum=momentum)

        print("loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        del checkpoint  # 删除载入的模型
        del model_dict
        print("加载已经保存好的模型")
    # resume 为false 训练图像深度模型
    else:
        print("创建模型")
        # 构建dorn网络初始化模型
        model = DORN()
        # torch 使用sgd构建优化器，sgd,Stochastic Gradient Descent,数据分批加载迭代，输入定义好模型初始化参数（变量上文已定义）
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=momentum)
        start_epoch = 0
    # 当前环境是否存在可加载的gpu cuda 资源
    if torch.cuda.device_count():
        print("当前GPU数量：", torch.cuda.device_count())
        # model = torch.nn.DataParallel(model)
    # 模型加载cuda
    model = model.cuda()
    # 定义损失函数
    criterion = ordLoss()
    # 初始化输出文件,如果文件不存在，创建对应文件夹和文件
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 记录模型迭代训练参数
    best_txt = os.path.join(output_dir, 'best.txt')
    # 记录模型日志
    log_path = os.path.join(output_dir, 'logs', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    os.makedirs(log_path)
    logger = SummaryWriter(log_path)

    # 开始训练
    for epoch in range(start_epoch, epoches):
        print("current epoch:" + str(epoch))
        # 模型训练
        train(train_loader, model, criterion, optimizer, epoch, logger)
        # 模型验证
        result, img_merge = validate(val_loader, model, epoch, logger)
        # 对比rmse,当前模型参数对应rmse是开始到当前迭代最佳
        is_best = result.rmse < best_result.rmse
        if is_best:
            # 是最佳则记录训练参数，并记录训练集，测试集和，验证集抽样图片深度对比图
            best_result = result
            with open(best_txt, 'w') as txtfile:
                txtfile.write(
                    "epoch={}\nrmse={:.3f}\nrml={:.3f}\nlog10={:.3f}\nd1={:.3f}\nd2={:.3f}\ndd31={:.3f}\nt_gpu={:.4f}\n".
                        format(epoch, result.rmse, result.absrel, result.lg10, result.delta1, result.delta2,
                               result.delta3,
                               result.gpu_time))
            if img_merge is not None:
                img_filename = output_dir + '/comparison_best.png'
                m_utils.save_image(img_merge, img_filename)

        # 每个epoch保存检查点
        save_checkpoint({'epoch': epoch, 'model': model, 'optimizer': optimizer, 'best_result': best_result},
                        is_best, epoch, output_dir)
        print("模型保存成功")


# 在NYU训练集上训练一个epoch
def train(train_loader, model, criterion, optimizer, epoch, logger):
    """
    模型训练定义函数
    :param train_loader: 训练集数据加载器
    :param model: 模型
    :param criterion: 决策树参数（参考信息熵）
    :param optimizer: 优化器
    :param epoch: 迭代轮次
    :param logger: 日志记录
    :return:
    """
    average_meter = AverageMeter()
    # 模型训练
    model.train()
    # 模型训练结束时间
    end = time.time()
    # 训练集批次数
    batch_num = len(train_loader)
    # 当前迭代步长
    current_step = batch_num * batch_size * epoch
    for i, (input, target) in enumerate(train_loader):
        # 学习率更新函数
        lr = update_ploy_lr(optimizer, init_lr, current_step, max_iter)
        if torch.cuda.is_available():
            # 如果cuda可用，获取输入和目标
            input, target = input.cuda(), target.cuda()
        # 学习率更新耗时
        data_time = time.time() - end

        current_step += input.data.shape[0]

        if current_step == max_iter:
            logger.close()
            print("迭代完成")
            break
        # cuda显存锁存，防止显存爆存
        torch.cuda.synchronize()
        end = time.time()
        # compute pred
        end = time.time()
        # 开启torch 自动求导
        with torch.autograd.detect_anomaly():
            pred_d, pred_ord = model(input)  # @wx 注意输出
            # 损失函数获取
            loss = criterion(pred_ord, target)
            # 参数梯度设置为0，防止维度爆炸
            optimizer.zero_grad()
            # 损失函数反向传播
            loss.backward()  # compute gradient and do SGD step
            # 优化器迭代
            optimizer.step()
        # 锁cuda
        torch.cuda.synchronize()

        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        # 获取深度结果
        depth = get_depth_sid(pred_d)
        target_dp = get_depth_sid(target)
        # 通过训练集和目标集差异评估模型
        result.evaluate(depth.data, target_dp.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        if (i + 1) % 10 == 0:
            # 输出对应每轮迭代参数
            print('Train Epoch: {0} [{1}/{2}]\t'
                  'learning_rate={lr:.8f} '
                  't_Data={data_time:.3f}({average.data_time:.3f}) '
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'Loss={loss:.3f} '
                  'RMSE={result.rmse:.3f}({average.rmse:.3f}) '
                  'RML={result.absrel:.3f}({average.absrel:.3f}) '
                  'Log10={result.lg10:.3f}({average.lg10:.3f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'Delta2={result.delta2:.3f}({average.delta2:.3f}) '
                  'Delta3={result.delta3:.3f}({average.delta3:.3f})'.format(
                epoch, i + 1, batch_num, lr=lr, data_time=data_time, loss=loss.item(),
                gpu_time=gpu_time, result=result, average=average_meter.average()))

            logger.add_scalar('Learning_rate', lr, current_step)
            logger.add_scalar('Train/Loss', loss.item(), current_step)
            logger.add_scalar('Train/RMSE', result.rmse, current_step)
            logger.add_scalar('Train/rml', result.absrel, current_step)
            logger.add_scalar('Train/Log10', result.lg10, current_step)
            logger.add_scalar('Train/Delta1', result.delta1, current_step)
            logger.add_scalar('Train/Delta2', result.delta2, current_step)
            logger.add_scalar('Train/Delta3', result.delta3, current_step)
        avg = average_meter.average()


def validate(val_loader, model, epoch, logger, write_to_file=True):
    """
    模型验证
    :param val_loader: 验证集加载器
    :param model: --
    :param epoch: --
    :param logger: --
    :param write_to_file: 写入文件名
    :return:
    """
    average_meter = AverageMeter()
    model.eval()
    end = time.time()
    # 模型验证和模型训练本质是同样的执行操作，同上文训练
    for i, (input, target) in enumerate(val_loader):
        if torch.cuda.is_available():
            input, target = input.cuda(), target.cuda()
        torch.cuda.synchronize()
        # 计算数据时间
        data_time = time.time() - end
        with torch.no_grad():
            pred_d, pred_ord = model(input)
        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # 度量
        result = Result()
        depth = get_depth_sid(pred_d)
        target_dp = get_depth_sid(target)
        result.evaluate(depth.data, target_dp.data)

        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        # 保存一些验证结果
        skip = 11
        rgb = input
        # 保留部分深度图
        if i == 0:
            img_merge = m_utils.merge_into_row(rgb, target_dp, depth)
        elif (i < 8 * skip) and (i % skip == 0):
            row = m_utils.merge_into_row(rgb, target_dp, depth)
            img_merge = m_utils.add_row(img_merge, row)
        elif i == 8 * skip:
            filename = output_dir + '/comparison_' + str(epoch) + '.png'
            m_utils.save_image(img_merge, filename)

        if (i + 1) % 10 == 0:
            print('Validate: [{0}/{1}]\t'
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'RML={result.absrel:.2f}({average.absrel:.2f}) '
                  'Log10={result.lg10:.3f}({average.lg10:.3f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'Delta2={result.delta2:.3f}({average.delta2:.3f}) '
                  'Delta3={result.delta3:.3f}({average.delta3:.3f})'.format(
                i + 1, len(val_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()

    print('\n*\n'
          'RMSE={average.rmse:.3f}\n'
          'Rel={average.absrel:.3f}\n'
          'Log10={average.lg10:.3f}\n'
          'Delta1={average.delta1:.3f}\n'
          'Delta2={average.delta2:.3f}\n'
          'Delta3={average.delta3:.3f}\n'
          't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))

    logger.add_scalar('Test/rmse', avg.rmse, epoch)
    logger.add_scalar('Test/Rel', avg.absrel, epoch)
    logger.add_scalar('Test/log10', avg.lg10, epoch)
    logger.add_scalar('Test/Delta1', avg.delta1, epoch)
    logger.add_scalar('Test/Delta2', avg.delta2, epoch)
    logger.add_scalar('Test/Delta3', avg.delta3, epoch)
    return avg, img_merge


if __name__ == '__main__':
    # 主函数入口
    main()
