import os
import json
import matplotlib.pyplot as plt


def plot_loss_curve(losses, save_path=None):
    """绘制训练损失曲线"""
    plt.figure()
    plt.plot(range(len(losses)), losses, label='Train Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


def save_args(args, save_dir):
    save_path = os.path.join(save_dir, 'args.txt')
    with open(save_path, 'w') as f:
        for k, v in vars(args).items():
            f.write(f'{k}: {v}\n')


def generate_readme(args, result_msg):
    """生成 ReadMe.txt，保存训练摘要信息"""
    readme_file = os.path.join(args.output_dir, 'ReadMe.txt')
    with open(readme_file, 'w') as f:
        lines = []
        lines.append(f'Train and valid with {args.input_file}\n')
        lines.append(f'Model: {args.model_name}\n')
        lines.append(result_msg)

        for line in lines:
            print(line)

        f.writelines(lines)


def get_last_net_dir(models_dir, model_name):
    """获取模型保存目录下最新的模型文件夹路径"""
    model_param_path = os.path.join(models_dir, model_name)
    model_save_time = None
    model_save_time = model_save_time \
        if model_save_time is not None else \
        sorted(
            list(filter(os.path.isdir, [os.path.join(
                model_param_path, x) for x in os.listdir(model_param_path)])),
            reverse=True)[3]
    return model_save_time


def plot_loss_lines(args, train_losses, valid_losses):
    """绘制训练与验证损失图"""
    plt.rcParams['font.sans-serif'] = ['SimHei']

    plot_interp_loss = False
    plot_kdm_loss = False

    interp_train_loss = 0.003315
    kdm_train_loss = 0.002543
    interp_valid_loss = 0.003321
    kdm_valid_loss = 0.0025

    interp_train_losses = [interp_train_loss] * len(train_losses)
    kdm_train_losses = [kdm_train_loss] * len(train_losses)
    interp_valid_losses = [interp_valid_loss] * len(valid_losses)
    kdm_valid_losses = [kdm_valid_loss] * len(valid_losses)

    # train loss
    figure_train = plt.figure()
    plt.plot(range(len(train_losses)), train_losses, color='r', label='SR')
    if plot_interp_loss:
        plt.plot(range(len(train_losses)), interp_train_losses,
                 color='b', linestyle='-.', label='interp')
    if plot_kdm_loss:
        plt.plot(range(len(train_losses)), kdm_train_losses,
                 color='g', linestyle='--', label='kdm')
    plt.legend()
    figure_train.savefig(os.path.join(
        args.output_dir, 'Train-Loss.png'), dpi=600)
    plt.close()

    # valid loss
    figure_valid = plt.figure()
    plt.plot(range(len(valid_losses)), valid_losses, color='r', label='SR')
    if plot_interp_loss:
        plt.plot(range(len(valid_losses)), interp_valid_losses,
                 color='b', linestyle='-.', label='interp')
    if plot_kdm_loss:
        plt.plot(range(len(valid_losses)), kdm_valid_losses,
                 color='g', linestyle='--', label='kdm')
    plt.legend()
    figure_valid.savefig(os.path.join(
        args.output_dir, 'Valid-Loss.png'), dpi=600)
    plt.close()


if __name__ == '__main__':
    pass
