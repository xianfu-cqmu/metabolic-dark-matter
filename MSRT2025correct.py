import os
import random
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from openpyxl import Workbook
import torch.nn.functional as F
import sys

sys.path.append('../fingerprint')
from PreAccCal import process_prediction
import time
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from utils.PreAccCal import process_prediction


def SetSeed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ========================
# 数据集定义 - 添加保留时间支持
# ========================
class SparseSpectralDataset(Dataset):
    def __init__(self, h5_file, label_file):
        self.h5_file = h5py.File(h5_file, 'r')
        labels = pd.read_excel(label_file)

        def parse_morgan_bits(bits_str):
            bit_indices = eval(bits_str)
            indices = torch.tensor(bit_indices, dtype=torch.long) - 1  # shape [N]
            values = torch.ones(len(bit_indices), dtype=torch.float32)
            return torch.sparse_coo_tensor(
                indices.unsqueeze(0), values, (2048,), dtype=torch.float32
            )

        self.labels = {
            str(row['CAS No.']): parse_morgan_bits(row['Morgan_Bits'])
            for _, row in labels.iterrows()
        }

        # 提取保留时间并标准化
        rt_values = [row['RT'] for _, row in labels.iterrows() if pd.notna(row['RT'])]
        self.rt_mean = np.mean(rt_values) if rt_values else 0.0
        self.rt_std = np.std(rt_values) if rt_values else 1.0

        # 防止 std = 0
        if self.rt_std < 1e-8:
            self.rt_std = 1.0

        self.rt_values = {
            str(row['CAS No.']): (row['RT'] - self.rt_mean) / self.rt_std
            for _, row in labels.iterrows() if pd.notna(row['RT'])
        }

        self.cas_nos = [key for key in self.h5_file.keys() if key in self.rt_values]
        self.samples = {key: self.h5_file[key][:] for key in self.cas_nos}

    def __len__(self):
        return len(self.cas_nos)

    def __getitem__(self, idx):
        cas_no = self.cas_nos[idx]
        sample = self.samples[cas_no]  # 单能量数据，形状应为 [1, 104000] 或 [104000,]

        # 确保样本是二维的 [1, 104000]
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)

        # 添加数据标准化
        min_val = np.min(sample)
        max_val = np.max(sample)
        if max_val > min_val:
            sample = (sample - min_val) / (max_val - min_val)

        row_idx, col_idx = np.nonzero(sample)
        if len(row_idx) == 0:
            sparse_sample = torch.zeros(sample.shape, dtype=torch.float32)
        else:
            indices = np.vstack((row_idx, col_idx))
            values = sample[row_idx, col_idx]
            sparse_sample = torch.sparse_coo_tensor(
                indices, values, sample.shape, dtype=torch.float32
            )

        label = self.labels[cas_no]
        rt = torch.tensor([self.rt_values[cas_no]], dtype=torch.float32)

        return sparse_sample, label.to_dense(), rt, cas_no


# ========================
# 批处理函数 - 添加保留时间支持
# ========================
def sparse_collate_fn(batch):
    samples, labels, rts, cas_nos = zip(*batch)
    dense_samples = torch.stack([x.to_dense() if x.is_sparse else x for x in samples])
    dense_labels = torch.stack(labels)
    dense_rts = torch.stack(rts)
    return dense_samples, dense_labels, dense_rts, list(cas_nos)


'''
# ========================

# 此处为基于理化属性约束定义的损失函数
# 代码在成果发表后公开


# ========================
'''


# 计算类别权重
# ========================
def calculate_class_weights(label_file):
    labels_df = pd.read_excel(label_file)
    all_bits = []

    for _, row in labels_df.iterrows():
        bits = eval(row['Morgan_Bits'])
        all_bits.extend(bits)

    # 计算每个位的频率
    bit_counts = np.bincount(all_bits, minlength=2049)[1:]  # 忽略0索引
    class_weights = 1.0 / (bit_counts + 1e-6)  # 添加小值避免除零
    class_weights = class_weights / np.sum(class_weights)  # 归一化

    return torch.tensor(class_weights, dtype=torch.float32)


# ========================
# 评估指标计算
# ========================
def calculate_metrics(preds, targets, threshold=0.5):
    preds_binary = (preds > threshold).float()

    # 计算准确率、精确率、召回率、F1
    tp = (preds_binary * targets).sum(dim=1)
    tn = ((1 - preds_binary) * (1 - targets)).sum(dim=1)
    fp = (preds_binary * (1 - targets)).sum(dim=1)
    fn = ((1 - preds_binary) * targets).sum(dim=1)

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    return {
        'accuracy': accuracy.mean(),
        'precision': precision.mean(),
        'recall': recall.mean(),
        'f1': f1.mean()
    }


'''
# ========================

# 多尺度和空洞卷积网络模型结构定义
# 此处代码在成果发表后公开


# ========================
'''

if __name__ == '__main__':

    start = time.time()
    SEED = 90
    SetSeed(SEED=SEED)

    # ========================
    # 模型保存路径
    # ========================
    save_dir = './OfflineModel'
    os.makedirs(save_dir, exist_ok=True)

    h5_file = "../data/targetC18neg204060revise.h5"
    label_file = "../fingerprint/MorganC18negRT.xlsx"

    # ========================
    # 构建数据加载器
    # ========================
    dataset = SparseSpectralDataset(h5_file, label_file)
    indices = torch.randperm(len(dataset)).tolist()
    train_indices, val_indices = indices[300:], indices[:300]

    train_set = torch.utils.data.Subset(dataset, train_indices)
    val_set = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=32,
        shuffle=True,
        collate_fn=sparse_collate_fn,
        generator=torch.Generator().manual_seed(SEED)
    )
    val_loader = DataLoader(
        val_set,
        batch_size=32,
        shuffle=False,
        collate_fn=sparse_collate_fn
    )

    # ========================
    # 初始化模型
    # ========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")
    model = MultiScaleDilatedCNN(fingerprint_dim=2048).to(device)

    '''
    # ========================

    # 使用保留时间加权的BCE损失（涉及核心参数设置）
    # 此处代码在成果发表后公开

    # ========================
    '''

    # 优化器和学习率调度器
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # ========================
    # 训练主循环
    # ========================
    epochs = 200
    best_val_loss = float('inf')

    # 添加调试信息
    print(f"Training samples: {len(train_set)}")
    print(f"Validation samples: {len(val_set)}")
    print(f"RT mean: {dataset.rt_mean}, RT std: {dataset.rt_std}")

    # 创建保存模型的目录
    os.makedirs('../Result/models', exist_ok=True)

    # 确定需要评估的epoch（从最大epoch倒数，以*为间隔，共10次）
    eval_epochs = []
    current_epoch = epochs
    for _ in range(10):
        if current_epoch <= 0:
            break
        eval_epochs.append(current_epoch)
        current_epoch -= 5

    if not eval_epochs:
        eval_epochs = [epochs]

    print(f"Model Will evaluate models at epochs: {eval_epochs}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for i, (x, y, rt, _) in enumerate(train_loader):
            x, y, rt = x.to(device), y.to(device), rt.to(device)

            # 检查输入数据
            if i == 0 and epoch == 0:
                print(f"Input shape: {x.shape}")
                print(f"Input non-zero elements: {torch.sum(x != 0)}")
                print(f"Label shape: {y.shape}")
                print(f"Label non-zero elements: {torch.sum(y != 0)}")
                print(f"RT shape: {rt.shape}")
                print(f"RT values: {rt[:5]}")

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y, rt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # 真正启用 scheduler
        scheduler.step(avg_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        # 只在需要评估的epoch保存模型
        if (epoch + 1) in eval_epochs:
            model_path = f'../Result/models/2025model_epoch_{epoch + 1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, model_path)
            print(f"💾 Model saved at epoch {epoch + 1}")

    print("Training completed.")

    # 进行模型评估
    for eval_epoch in eval_epochs:
        try:
            # 加载对应epoch的模型
            model_path = f'../Result/models/2025model_epoch_{eval_epoch}.pth'
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"🔍 Evaluating model from epoch {eval_epoch}")

            model.eval()
            results = []
            with torch.no_grad():
                for x, y, rt, cas_nos in val_loader:
                    x = x.to(device)
                    out = model(x)
                    pred_probs = torch.sigmoid(out)
                    preds = (pred_probs > 0.5).int().cpu().numpy()

                    for cas, pred in zip(cas_nos, preds):
                        bits = np.where(pred == 1)[0] + 1
                        results.append([cas, str(bits.tolist())])

            # 保存预测结果，文件名中包含epoch信息
            save_path = f"../Result/2025Prediction_C18neg_204060byT3seed100RTWeightedEpoch{eval_epoch}.xlsx"
            try:
                df = pd.DataFrame(results, columns=["CAS No.", "Predicted Morgan Fingerprint"])
                df.to_excel(save_path, index=False)
                print(f"✅ Successfully saved predictions for epoch {eval_epoch} to Excel file")
            except Exception as e:
                print(f"❌ Error saving Excel file for epoch {eval_epoch}: {e}")
                continue

            # 处理预测结果
            PreResult = process_prediction(save_path)

            # 记录结果，包含epoch信息
            with open('../Result/AccResult.txt', 'a') as file:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"[{timestamp}] Epoch {eval_epoch}: {PreResult['percentage']}\n")

            print(f"📈 Epoch {eval_epoch} evaluation completed with result: {PreResult['percentage']}")

        except FileNotFoundError:
            print(f"⚠️  Model file for epoch {eval_epoch} not found, skipping...")
        except Exception as e:
            print(f"❌ Error evaluating model from epoch {eval_epoch}: {e}")

    print("All evaluations completed.")

    end = time.time()
    print(f"程序运行时间：{(end - start) / 60: .2f}分钟")
