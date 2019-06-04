import config
import torch
from allennlp.nn.util import move_to_device

# init params
nelConfig = config.NELConfig()
comConfig = config.ComConfig()


def get_accuracy(scores, target):
    '''
    :param scores: batch, cands
    :param target: batch
    :return:
    '''
    value, position = torch.max(scores, dim=-1)
    return torch.sum(position == target).float() / scores.size(0)


def train_eval(model, train_iter, dev_iter, opt):
    step = 0
    for epoch in range(nelConfig.train_epoches):
        model.train()
        model.to(comConfig.device)
        train_loss = []
        train_acc = []
        for batch in train_iter:
            batch = move_to_device(batch, 0)
            mention_context = batch['mention_context']
            mention_position = batch['mention_position']
            entity_cands = batch['entity_cands_id']
            entity_context = batch['entity_contexts_id']
            target = batch['target']
            opt.zero_grad()
            scores, loss = model(mention_context, mention_position, entity_context, entity_cands, target)
            train_acc.append(get_accuracy(scores, target))
            loss.backward()
            train_loss.append(loss.item())
            opt.step()
            step += 1
            if step % 10 == 0:
                # torch.save(model.state_dict(), os.path.join('entity_model', str(step) + '.pt'))
                print("train iter epoch:{}/{} step:{} loss:{:.4f} acc:{:.4f}".format(epoch, nelConfig.train_epoches, step,
                                                                             sum(train_loss) / len(train_loss),
                                                                             sum(train_acc) / len(train_acc)))
        model.eval()
        eval_loss = []
        eval_acc = []
        for batch in dev_iter:
            batch = move_to_device(batch, 0)
            mention_context = batch['mention_context']
            mention_position = batch['mention_position']
            entity_cands = batch['entity_cands_id']
            entity_context = batch['entity_contexts_id']
            target = batch['target']
            scores, loss = model(mention_context, mention_position, entity_context, entity_cands, target)

            eval_loss.append(loss.item())
            eval_acc.append(get_accuracy(scores, target))
        print("eval iter epoch:{}/{} loss:{:.4f} acc:{:.4f}".format(epoch, nelConfig.train_epoches,
                                                            sum(eval_loss) / len(eval_loss),
                                                            sum(eval_acc) / len(eval_acc)))
        train_loss = []
        train_acc = []
        epoch += 1
