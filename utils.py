import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


# 追加（apply_crf関数までの記述）
# import pydensecrf.densecrf as dcrf
# from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
# import copy
# def apply_crf(image, segmentation_prob, sxy_gaussian=3, compat_gaussian=3, sxy_bilateral=80, srgb_bilateral=13, compat_bilateral=10):
#     # 画像の形状を取得
#     h, w = image.shape[:2]

#     # DenseCRFの初期化
#     d = dcrf.DenseCRF2D(w, h, segmentation_prob.shape[0])

#     # Unaryエネルギーを設定
#     unary = unary_from_softmax(segmentation_prob)
#     d.setUnaryEnergy(unary)

#     # Gaussianのペアワイズタームを追加
#     d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian)

#     # Bilateralのペアワイズタームを追加（色を考慮）
#     d.addPairwiseBilateral(sxy=sxy_bilateral, srgb=srgb_bilateral, rgbim=image, compat=compat_bilateral)

#     # CRFを実行
#     Q = d.inference(5)  # 反復回数を設定

#     # CRF後のラベルを取得
#     return np.argmax(Q, axis=0).reshape((h, w))

def apply_crf(image, prediction, num_classes=2, spatial_sigma=3, bilateral_sigma_color=10, bilateral_sigma_spatial=3):
    """
    グレースケール画像に対するCRFの適用。
    
    image: 元のグレースケール画像（2D配列）
    prediction: セグメンテーション結果（2D配列）
    num_classes: クラス数（デフォルトは2クラス）
    spatial_sigma: ガウシアンフィルタの空間シグマ
    bilateral_sigma_color: バイラテラルフィルタの色シグマ
    bilateral_sigma_spatial: バイラテラルフィルタの空間シグマ
    """
    
    # CRFモデルの作成
    h, w = image.shape
    d = dcrf.DenseCRF2D(w, h, num_classes)
    
    # ユニアリティ項を設定（セグメンテーション予測から作成）
    labels = prediction.astype(np.int32)
    unary = unary_from_labels(labels, num_classes, gt_prob=0.7)
    unary = unary.reshape((num_classes, -1))  # (クラス数, ピクセル数)に変換
    d.setUnaryEnergy(unary)
    
    # ペアワイズガウシアン項を追加（スムージング効果）
    gaussian_energy = create_pairwise_gaussian(sdims=(spatial_sigma, spatial_sigma), shape=image.shape)
    d.addPairwiseEnergy(gaussian_energy, compat=3)
    
    # ペアワイズバイラテラル項を追加（強度値を考慮したスムージング）
    bilateral_energy = create_pairwise_bilateral(sdims=(bilateral_sigma_spatial, bilateral_sigma_spatial),
                                                 schan=(bilateral_sigma_color,),
                                                 img=image,
                                                 chdim=0)
    d.addPairwiseEnergy(bilateral_energy, compat=10)
    
    # 反復回数を指定してCRFを実行
    Q = d.inference(5)
    
    # 各ピクセルのクラスラベルを取得
    result = np.argmax(Q, axis=0).reshape((h, w))
    
    return result



def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()

    # 追加（predictionまでの記述）
    # --- CRF適用前のpredictionを保存 ---
    original_prediction = copy.deepcopy(prediction)

    # --- CRFの適用 ---
    prediction = apply_crf(image, prediction)
    
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    # 変更（metric_list -> metric_list, original_prediction, prediction）
    return metric_list, original_prediction, prediction
