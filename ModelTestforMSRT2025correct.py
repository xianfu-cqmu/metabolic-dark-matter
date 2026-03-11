import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py
import pandas as pd
from openpyxl import Workbook
from MSRT2025correct import MultiScaleDilatedCNN   # 把训练脚本文件名改成这个更稳

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


class SparseSpectralDataset(Dataset):
    def __init__(self, h5_file, label_file=None, mode='train'):
        """
        mode: 'train' 或 'test'
        如果 mode='train'，需要传入 label_file 并返回 (sample, label, cas_no)
        如果 mode='test'，忽略 label_file，只返回 (sample, cas_no)
        """
        self.mode = mode
        self.h5_file = h5py.File(h5_file, 'r')
        self.cas_nos = list(self.h5_file.keys())

        # 只在 train 模式下读取并解析 labels
        if self.mode == 'train':
            assert label_file is not None, "Train mode requires label_file"
            labels_df = pd.read_excel(label_file)

            def parse_morgan_bits(bits_str):
                bit_indices = eval(bits_str)
                indices = torch.tensor([bit_indices], dtype=torch.long) - 1
                values = torch.ones(len(bit_indices), dtype=torch.float32)
                return torch.sparse_coo_tensor(indices, values, (2048,), dtype=torch.float32)

            self.labels = {
                str(row['CAS No.']): parse_morgan_bits(row['Morgan_Bits'])
                for _, row in labels_df.iterrows()
            }

    def __len__(self):
        return len(self.cas_nos)

    def __getitem__(self, idx):
        cas_no = self.cas_nos[idx]
        sample_np = self.h5_file[cas_no][:]

        # 与训练保持一致：如果是一维，变成 [1, 104000]
        if sample_np.ndim == 1:
            sample_np = sample_np.reshape(1, -1)

        # 与训练保持一致：min-max 归一化
        min_val = np.min(sample_np)
        max_val = np.max(sample_np)
        if max_val > min_val:
            sample_np = (sample_np - min_val) / (max_val - min_val)

        # 构造稀疏张量
        non_zero = np.nonzero(sample_np)
        indices = torch.tensor(np.vstack(non_zero), dtype=torch.int64)
        values = torch.tensor(sample_np[non_zero], dtype=torch.float32)
        sparse_sample = torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=sample_np.shape,
            dtype=torch.float32
        )

        if self.mode == 'train':
            label = self.labels[cas_no].to_dense()
            return sparse_sample, label, cas_no
        else:  # test 模式
            return sparse_sample, cas_no


def sparse_collate_fn(batch):
    """
    A collate function that handles both train (sample, label, cas_no)
    and test (sample, cas_no) modes automatically.
    """
    first = batch[0]

    if len(first) == 3:
        # TRAIN 模式
        sparse_samples = []
        dense_labels = []
        cas_nos = []

        for sample, label, cas_no in batch:
            sparse_samples.append(sample)
            dense_labels.append(label)
            cas_nos.append(cas_no)

        sparse_batch = torch.stack([x.to_dense() for x in sparse_samples])
        dense_labels = torch.stack(dense_labels)

        return sparse_batch, dense_labels, cas_nos

    elif len(first) == 2:
        # TEST 模式
        sparse_samples = []
        cas_nos = []

        for sample, cas_no in batch:
            sparse_samples.append(sample)
            cas_nos.append(cas_no)

        sparse_batch = torch.stack([x.to_dense() for x in sparse_samples])
        return sparse_batch, cas_nos

    else:
        raise RuntimeError(f"Unexpected batch element size: {len(first)}")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 测试HDF5文件所在路径，并进行读取
    TestFile = "../data/GBMnegtestspectra_revise.h5"
    test_ds = SparseSpectralDataset(TestFile, mode='test')
    Test_loader = DataLoader(
        test_ds,
        batch_size=64,
        shuffle=False,
        collate_fn=sparse_collate_fn
    )
    print('数据加载完毕！')

    # 1. 创建模型实例
    model = MultiScaleDilatedCNN().to(device)

    # 2. 加载权重
    checkpoint = torch.load("../Result/models/2025model_epoch_200.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('模型加载完毕！')

    # 3. 推理
    model.eval()
    results = []

    with torch.no_grad():
        for samples, cas_nos in Test_loader:
            samples = samples.to(device)
            outputs = model(samples)
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > 0.5).int().cpu().numpy()

            for cas_no, output in zip(cas_nos, outputs):
                # 将预测结果（1的位置转换为索引列表）并加1恢复到原索引编号
                non_zero_indices = np.where(output == 1)[0] + 1
                results.append([cas_no, str(non_zero_indices.tolist())])

    # 4. 保存预测结果到 Excel
    save_path = '../Result/Test_GBMneg.xlsx'
    wb = Workbook()
    ws = wb.active
    ws.append(["CAS No.", "Predicted Morgan Fingerprint"])
    for result in results:
        ws.append(result)
    wb.save(save_path)

    print(f'预测完成，结果已保存到: {save_path}')