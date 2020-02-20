import os
import matplotlib.pyplot as plt


def plot_result(args, train_loss, train_accuracy, test_loss, test_accuracy, save_plot=True):
    xs = list(range(len(train_loss)))
    f, (fg1, fg2) = plt.subplots(1, 2)
    fg1.set_title('Loss during training')
    fg1.plot(xs, train_loss, '-b',label='Train')
    fg1.plot(xs, test_loss, '-r',label='Test')
    fg1.set_xlabel ('Epochs')
    fg1.set_ylabel('Loss')

    fg1.legend()

    fg2.set_title('Accuracy during training')
    fg2.plot(xs, train_accuracy, '-b',label='Train')
    fg2.plot(xs, test_accuracy, '-r',label='Test')
    fg2.set_xlabel ('Epochs')
    fg2.set_ylabel('Accuracy [%]')

    fg2.legend()
    if save_plot:
        plot_save_path = os.path.join(args.out_dir, 'figs', 'loss_plot_{}.png'.format(args.experiment_id))
        os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
        plt.savefig(plot_save_path)
        plt.close()
    else:
        plt.show()
