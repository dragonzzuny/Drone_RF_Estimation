"""Full GPU equivalence test of direct and nested-checkpoint backward."""
import torch
from torch.utils.checkpoint import checkpoint


def checkpoint_predict(net,item,predict):
    def forward(mix,context,position):
        return predict(net,dict(mixture=mix,context_features=context,crop_start=position))
    return checkpoint(forward,item['mixture'],item['context_features'],item['crop_start'],use_reentrant=False)


def preflight(net,item,predict,objective):
    net.train();net.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
    assert torch.count_nonzero(net.septda.source_readout.weight)==0
    try:
        # Enable the new internal paths temporarily; zero readout would hide
        # internal-gradient differences. No optimizer step or retained change.
        with torch.no_grad():
            torch.manual_seed(17);torch.nn.init.normal_(net.septda.source_readout.weight,std=1e-4)
        output,logits=predict(net,item);loss=objective(output,logits);loss.backward()
        assert all(q.grad is not None and bool(torch.isfinite(q.grad).all()) for q in net.parameters())
        output_ref=output.detach().cpu();logits_ref=logits.detach().cpu();loss_ref=float(loss.detach())
        gradients={name:q.grad.detach().cpu().clone() for name,q in net.named_parameters()}
        assert float(gradients['septda.queries'].norm())>0
        del output,logits,loss;net.zero_grad(set_to_none=True)
        output,logits=checkpoint_predict(net,item,predict);loss=objective(output,logits);loss.backward()
        torch.testing.assert_close(output.detach().cpu(),output_ref,rtol=1e-6,atol=1e-7)
        torch.testing.assert_close(logits.detach().cpu(),logits_ref,rtol=1e-6,atol=1e-7)
        error=reference=0.;maximum=0.;exact=True
        for name,q in net.named_parameters():
            assert q.grad is not None and bool(torch.isfinite(q.grad).all())
            actual=q.grad.detach().cpu();expected=gradients[name]
            torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-7)
            delta=(actual-expected).double();error+=float(delta.square().sum());reference+=float(expected.double().square().sum())
            maximum=max(maximum,float(delta.abs().max()));exact=exact and torch.equal(actual,expected)
        relative=(error/max(reference,1e-30))**.5;assert relative<1e-5
        assert abs(float(loss.detach())-loss_ref)<1e-6
        return dict(status='PASS',parameters=sum(q.numel() for q in net.parameters()),
            peak_bytes=torch.cuda.max_memory_allocated(),loss=loss_ref,optimizer_steps=0,
            nested_checkpoint_checked=True,gradient_relative_l2=relative,max_abs_gradient_difference=maximum,
            gradients_bitwise_equal=exact,enabled_query_gradient_norm=float(gradients['septda.queries'].norm()),
            source_readout_restored_zero=True)
    finally:
        with torch.no_grad():net.septda.source_readout.weight.zero_()
        net.zero_grad(set_to_none=True)
