"""Two successive full-capacity U-Net passes, three additive I/Q outputs.

This is an untrained architecture candidate, not OR-PIT/DExFormer reproduction.
Each pass chooses the highest-energy of the pretrained three source slots.
The original long mixture context conditions both passes deliberately. It is
not presented as a recomputed long-residual context. No target enters forward.
"""
import torch


def strongest_source(predictions):
    if predictions.ndim!=3 or predictions.shape[1]!=4 or not predictions.is_complex():
        raise ValueError('Expected three complex source slots and one background')
    if not bool(torch.isfinite(predictions).all()):raise FloatingPointError('Nonfinite prediction')
    index=predictions[:,:3].abs().square().mean(-1).argmax(-1)
    chosen=predictions.gather(1,index[:,None,None].expand(-1,1,predictions.shape[-1]))[:,0]
    return chosen,index


def predict(net,batch,base_predict):
    # Whitelist excludes references, classes, and true source count.
    original={key:batch[key] for key in ('mixture','context_features','crop_start')}
    first,logits=base_predict(net,original)
    s1,slot1=strongest_source(first)
    remaining=original['mixture']-s1
    # Do not detach: final reconstruction loss must reach the first extraction
    # through the second network's input as well as through direct outputs.
    second_input=dict(original,mixture=remaining)
    second,_=base_predict(net,second_input)
    s2,slot2=strongest_source(second)
    s3=remaining-s2
    output=torch.stack((s1,s2,s3,torch.zeros_like(s3)),1)
    return output,logits,dict(first_slot=slot1,second_slot=slot2,remaining=remaining,
                             first_predictions=first)
