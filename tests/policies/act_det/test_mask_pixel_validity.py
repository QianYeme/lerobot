"""Exercise ignored pixels through loss, NPZ loader, and actual policy forward."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lerobot.policies.act_det.detection.mask_decoder import mask_supervision_loss
from lerobot.policies.act_det.mask_loader import MaskLoader
from test_mask_supervision import MaskSupervisionTests


class PixelValidityTests(unittest.TestCase):
    def test_masked_bce_dice_matches_selected_pixel_formula(self):
        prediction = torch.tensor([[[[0.2,-0.4],[8.0,-8.0]]]], requires_grad=True)
        target = torch.tensor([[[[1.0,0.0],[0.0,1.0]]]])
        valid = torch.tensor([[[[True,True],[False,False]]]])
        p = prediction[valid].sigmoid()
        t = target[valid]
        expected = F.binary_cross_entropy_with_logits(prediction[valid],t)
        expected += 1-(2*(p*t).sum()+1e-6)/(p.sum()+t.sum()+1e-6)
        torch.testing.assert_close(mask_supervision_loss(prediction,target,"bce_dice",valid),expected)

    def test_ignored_predictions_and_targets_do_not_change_loss_or_gradients(self):
        valid = torch.tensor([[[[True,False],[True,False]]]])
        target = torch.ones(1,1,2,2)
        for kind in ["l1","bce_dice"]:
            prediction = torch.full_like(target,0.3,requires_grad=True)
            loss = mask_supervision_loss(prediction,target,kind,valid)
            changed = prediction.detach().clone()
            changed[~valid] = 100 if kind == "bce_dice" else 0.9
            changed_target = target.clone(); changed_target[~valid] = 0
            torch.testing.assert_close(loss,mask_supervision_loss(changed,changed_target,kind,valid))
            loss.backward()
            self.assertTrue((prediction.grad[~valid] == 0).all())
            self.assertGreater(prediction.grad[valid].abs().sum().item(),0)

    def test_all_ignored_loss_and_gradients_are_zero(self):
        for kind in ["l1","bce_dice"]:
            prediction = torch.zeros(2,1,4,5,requires_grad=True)
            loss = mask_supervision_loss(prediction,torch.ones_like(prediction),kind,torch.zeros_like(prediction,dtype=torch.bool))
            self.assertEqual(loss.item(),0)
            loss.backward()
            self.assertTrue((prediction.grad == 0).all())

    def test_duplicate_batch_and_empty_frames_preserve_loss(self):
        prediction = torch.zeros(1,1,4,5)
        target = torch.ones_like(prediction)
        valid = torch.ones_like(prediction,dtype=torch.bool); valid[:,:,:,3:] = False
        for kind in ["l1","bce_dice"]:
            expected = mask_supervision_loss(prediction,target,kind,valid)
            actual = mask_supervision_loss(prediction.repeat(3,1,1,1),target.repeat(3,1,1,1),kind,valid.repeat(3,1,1,1))
            torch.testing.assert_close(actual,expected)
            actual = mask_supervision_loss(prediction.repeat(2,1,1,1),target.repeat(2,1,1,1),kind,
                                           torch.cat([valid,torch.zeros_like(valid)]))
            torch.testing.assert_close(actual,expected)

    def test_full_resolution_half_precision_is_finite(self):
        torch.set_num_threads(2)
        prediction = torch.zeros(1,1,480,640,dtype=torch.float16,requires_grad=True)
        target = torch.ones_like(prediction)
        valid = torch.ones_like(prediction,dtype=torch.bool)
        actual = mask_supervision_loss(prediction,target,"bce_dice",valid)
        expected = mask_supervision_loss(prediction.float(),target.float(),"bce_dice",valid)
        self.assertTrue(torch.isfinite(actual))
        torch.testing.assert_close(actual.float(),expected,rtol=1e-4,atol=1e-4)
        actual.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(prediction.grad.abs().sum().float().item(),0)

    def test_loader_pixel_validity_legacy_fallback_and_eviction(self):
        with tempfile.TemporaryDirectory() as directory:
            top = Path(directory)/"top"; top.mkdir()
            masks = np.zeros((2,4,5),dtype=np.uint8)
            pixels = np.ones_like(masks,dtype=bool); pixels[:,:,2:] = False
            np.savez_compressed(top/"episode_000.npz",masks=masks,valid_pixels=pixels,valid=[True,False])
            np.savez_compressed(top/"episode_001.npz",masks=masks)
            loader = MaskLoader(directory,strict=True,max_cache_episodes=1)
            mask,valid = loader.get_mask_and_valid_pixels("observation.images.top",0,0)
            np.testing.assert_array_equal(valid,pixels[0]); self.assertEqual(mask.shape,(4,5))
            self.assertEqual(loader.get_mask_and_valid_pixels("observation.images.top",0,1),(None,None))
            _,valid = loader.get_mask_and_valid_pixels("observation.images.top",1,0)
            self.assertTrue(valid.all())
            _,valid = loader.get_mask_and_valid_pixels("observation.images.top",0,0)
            np.testing.assert_array_equal(valid,pixels[0])

    def test_invalid_pixel_arrays_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            top = Path(directory)/"top"; top.mkdir()
            for episode,pixels in enumerate([np.ones((1,2,2)),np.full((1,4,5),0.5),np.full((1,4,5),np.nan)]):
                np.savez_compressed(top/f"episode_{episode:03d}.npz",masks=np.zeros((1,4,5)),valid_pixels=pixels)
            loader = MaskLoader(directory,strict=True)
            for episode in range(3):
                with self.assertRaises(ValueError):
                    loader.get_mask_and_valid_pixels("observation.images.top",episode,0)

    def fixture(self):
        fixture = MaskSupervisionTests("test_inference_needs_neither_masks_nor_decoder")
        fixture.setUp(); self.addCleanup(fixture.tearDown)
        fixture.policy.config.mask_loss_type = "bce_dice"
        return fixture

    def test_policy_forward_counts_pixels_and_masks_decoder_output_gradients(self):
        fixture = self.fixture()
        masks = np.zeros((2,96,128),dtype=np.uint8); masks[:,24:60,40:80] = 1
        valid = np.zeros_like(masks,dtype=bool); valid[0,30:32,50:52] = True
        np.savez_compressed(fixture.mask_dir/"top/episode_000.npz",masks=masks,valid_pixels=valid)
        predictions = []
        def hook(module,inputs,output):
            output.retain_grad(); predictions.append(output)
        handle = fixture.policy.model.mask_decoder.register_forward_hook(hook)
        _,logs = fixture.policy(fixture.batch(2,[0,1])); handle.remove()
        self.assertEqual(logs["mask_valid_pixels"],4)
        self.assertAlmostEqual(logs["mask_coverage"],4/(2*96*128))
        fixture.policy.model.get_mask_loss()["mask_loss"].backward()
        self.assertTrue((predictions[0].grad[~torch.from_numpy(valid[:,None])] == 0).all())
        self.assertGreater(predictions[0].grad.abs().sum().item(),0)

    def test_policy_all_ignored_keeps_action_training_active(self):
        fixture = self.fixture()
        masks = np.zeros((2,96,128),dtype=np.uint8)
        np.savez_compressed(fixture.mask_dir/"top/episode_000.npz",masks=masks,valid_pixels=np.zeros_like(masks,dtype=bool))
        loss,logs = fixture.policy(fixture.batch())
        self.assertEqual(logs["mask_valid_pixels"],0); self.assertEqual(logs["mask_loss"],0)
        self.assertTrue(torch.isfinite(loss)); loss.backward()
        self.assertGreater(fixture.policy.model.action_head.weight.grad.abs().sum().item(),0)
        for parameter in fixture.policy.model.mask_decoder.parameters():
            self.assertTrue(parameter.grad is None or (parameter.grad == 0).all())


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PixelValidityTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
