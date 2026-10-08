import unittest
import torch
from phase_core import align_predictions,phase_predictions


class PhaseTests(unittest.TestCase):
    def test_whole_window_prediction_matching(self):
        torch.manual_seed(4)
        anchor=torch.randn(1,4,256,dtype=torch.complex64)
        matched,order=align_predictions(anchor[:,[2,0,1,3]],anchor)
        self.assertTrue(torch.equal(matched,anchor));self.assertEqual(order,[1,2,0])

    def test_equivariant_separator_unchanged_and_labels_blocked(self):
        torch.manual_seed(6)
        x=torch.randn(1,256,dtype=torch.complex64)
        def predict(model,item):
            self.assertNotIn('references',item)
            self.assertNotIn('construction_count',item)
            w=torch.tensor([.1,.2,.3,.4])[None,:,None]
            return item['mixture'][:,None]*w,torch.zeros(1,3)
        item=dict(mixture=x,context_features=torch.zeros(1,65,256),crop_start=torch.zeros(1),
            references='forbidden',construction_count='forbidden')
        full,average,_,receipt=phase_predictions(None,predict,item)
        self.assertTrue(torch.equal(full,average))
        self.assertTrue(torch.allclose(average.sum(1),x))
        self.assertEqual(receipt['inverse_rotated_prediction_relative_changes'],[0.,0.,0.,0.])

    def test_perfect_one_source_input_keeps_case_with_undefined_gain(self):
        from run_phase import row_metrics
        x=torch.tensor([1.,-1.,1.,-1.],dtype=torch.complex64)[None]
        estimate=torch.zeros(1,4,4,dtype=torch.complex64)
        estimate[:,0]=.75*x+torch.tensor([.1,.1,-.1,-.1])
        estimate[:,3]=x-estimate[:,0]
        references=torch.zeros(1,3,4,dtype=torch.complex64);references[:,0]=x
        item=dict(mixture=x,references=references,active=torch.tensor([[True,False,False]]),
            construction_count=torch.tensor([1]))
        row=row_metrics(estimate,torch.zeros(1,3),item,dict(levels=torch.zeros(3)),
            [dict(category='synthetic',pack_id='unit')],0,{})
        self.assertEqual(row['input_si_sdr'],[None]);self.assertEqual(row['si_sdr_gain'],[None])
        self.assertTrue(torch.isfinite(torch.tensor(row['nmse'])).all())
        self.assertTrue(torch.isfinite(torch.tensor(row['si_sdr'])).all())
        self.assertLess(row['sum_relative_error'],1e-9)


if __name__=='__main__':
    unittest.main()
