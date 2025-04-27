import time
import torch
import torch.nn.functional as F

def train_model(model, opt):
    print("training model...")
    model.train()
    start = time.time()
    cptime = start

    for epoch in range(opt.epochs):

        total_loss = 0

        for i, batch in enumerate(opt.train):

            src = batch.src.transpose(0, 1)
            trg = batch.trg.transpose(0, 1)
            trg_input = trg[:, :-1]
            src_mask, trg_mask = None, None
            preds = model(src, trg_input, src_mask, trg_mask)
            ys = trg[:, 1:].contiguous().view(-1)
            opt.optimizer.zero_grad()
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=opt.trg_pad)
            loss.backward()
            opt.optimizer.step()

            total_loss += loss.item()

            if (i + 1) % opt.printevery == 0:
                p = int(100 * (i + 1) / opt.train_len)
                avg_loss = total_loss / opt.printevery
                if opt.floyd is False:
                    print("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f" % \
                          ((time.time() - start) // 60, epoch + 1, "".join('#' * (p // 5)),
                           "".join(' ' * (20 - (p // 5))), p, avg_loss), end='\r')
                else:
                    print("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f" % \
                          ((time.time() - start) // 60, epoch + 1, "".join('#' * (p // 5)),
                           "".join(' ' * (20 - (p // 5))), p, avg_loss))
                total_loss = 0

            if opt.checkpoint > 0 and ((time.time() - cptime) // 60) // opt.checkpoint >= 1:
                torch.save(model.state_dict(), 'weights/model_weights')
                cptime = time.time()

        print("%dm: epoch %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, loss = %.03f" % \
              ((time.time() - start) // 60, epoch + 1, "".join('#' * (100 // 5)), "".join(' ' * (20 - (100 // 5))), 100,
               avg_loss, epoch + 1, avg_loss))