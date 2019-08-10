import torch
import config

from utils import com_utils
from utils import plot_utils
from pytorch_pretrained_bert.optimization import BertAdam

# init params
fileConfig = config.FileConfig()
comConfig = config.ComConfig()
nerConfig = config.NERConfig()
torch.manual_seed(comConfig.random_seed)
torch.cuda.manual_seed(comConfig.random_seed)
torch.cuda.manual_seed_all(comConfig.random_seed)


def fit_eval(model, training_iter, eval_iter, num_epoch, pbar, num_train_steps, verbose=1):
    # ---------------------优化器-------------------------
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    t_total = num_train_steps
    ## ---------------------GPU半精度fp16-----------------------------
    if nerConfig.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=nerConfig.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if nerConfig.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=nerConfig.loss_scale)
    ## ------------------------GPU单精度fp32---------------------------
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=nerConfig.learning_rate,
                             warmup=nerConfig.warmup_proportion,
                             t_total=t_total, weight_decay=0.1)
    # ---------------------模型初始化----------------------
    if nerConfig.fp16:
        model.half()
    model.to(comConfig.device)
    # use for plot
    train_losses = []
    eval_losses = []
    train_accuracy = []
    eval_accuracy = []
    history = {
        "train_loss": train_losses,
        "train_acc": train_accuracy,
        "eval_loss": eval_losses,
        "eval_acc": eval_accuracy}
    # ------------------------训练------------------------------
    print("start train model...")
    best_f1 = 0
    start = com_utils.get_time()
    global_step = 0
    print_count = 0
    for e in range(num_epoch):
        model.train()
        for step, batch in enumerate(training_iter):
            batch = tuple(t.to(comConfig.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, output_mask, input_lengths = batch
            bert_encode = model(input_ids, segment_ids, input_mask, input_length=input_lengths)
            train_loss = model.loss_fn(bert_encode=bert_encode, tags=label_ids, output_mask=output_mask)
            if nerConfig.gradient_accumulation_steps > 1:
                train_loss = train_loss / nerConfig.gradient_accumulation_steps
            if nerConfig.fp16:
                optimizer.backward(train_loss)
            else:
                train_loss.backward()
            if (step + 1) % nerConfig.gradient_accumulation_steps == 0:
                # modify learning rate with special warm up BERT uses
                lr_this_step = nerConfig.learning_rate * warmup_linear(global_step / t_total,
                                                                       nerConfig.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            predicts = model.predict(bert_encode, output_mask)
            label_ids = label_ids.view(1, -1)
            label_ids = label_ids[label_ids != -1]
            label_ids = label_ids
            train_acc, f1 = model.acc_f1(predicts, label_ids)
            # show train process
            print_count += 1
            if print_count % 10 == 0:
                pbar.show_process(train_acc, train_loss.item(), f1, com_utils.get_time() - start, step)
                print_count = 0
        # -----------------------验证----------------------------
        print('start eval model...')
        model.eval()
        count = 0
        y_predicts, y_labels = [], []
        eval_loss, eval_acc, eval_f1 = 0, 0, 0
        with torch.no_grad():
            for step, batch in enumerate(eval_iter):
                batch = tuple(t.to(comConfig.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, output_mask, eval_input_length = batch
                bert_encode = model(input_ids, segment_ids, input_mask, input_length=eval_input_length).to(
                    comConfig.device)
                eval_los = model.loss_fn(bert_encode=bert_encode, tags=label_ids, output_mask=output_mask)
                eval_loss = eval_los + eval_loss
                count += 1
                predicts = model.predict(bert_encode, output_mask)
                y_predicts.append(predicts)
                label_ids = label_ids.view(1, -1)
                label_ids = label_ids[label_ids != -1]
                y_labels.append(label_ids)
            eval_predicted = torch.cat(y_predicts, dim=0).to(comConfig.device)
            eval_labeled = torch.cat(y_labels, dim=0).to(comConfig.device)
            eval_acc, eval_f1 = model.acc_f1(eval_predicted, eval_labeled)
            model.class_report(eval_predicted, eval_labeled)
            print('\n\nEpoch %d - train_loss: %4f - eval_loss: %4f - train_acc:%4f - eval_acc:%4f - eval_f1:%4f\n' % (
                e + 1, train_loss.item(), eval_loss.item() / count, train_acc, eval_acc, eval_f1))
            # 保存最好的模型
            if eval_f1 > best_f1:
                best_f1 = eval_f1
                com_utils.save_model(model, fileConfig.dir_ner_checkpoint)
            if e % verbose == 0:
                train_losses.append(train_loss.item())
                train_accuracy.append(train_acc)
                eval_losses.append(eval_loss.item() / count)
                eval_accuracy.append(eval_acc)
    plot_utils.loss_acc_plot(history)


def predict(model, predict_iter):
    print('start predict dev data...')
    model.to(comConfig.device)
    model.eval()
    y_predicts = []
    with torch.no_grad():
        for step, batch in enumerate(predict_iter):
            batch = tuple(t.to(comConfig.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, output_mask, input_length = batch
            bert_encode = model(input_ids, segment_ids, input_mask).to(comConfig.device)
            predicts = model.predict(bert_encode, output_mask, is_squeeze=False)
            y_predicts.append(predicts)
    return y_predicts


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x
