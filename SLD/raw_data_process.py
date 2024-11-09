
import pandas as pd
import numpy as np
import torch

order_df = pd.read_excel(r"order.xlsx")
supply_df = pd.read_excel(r"supply.xlsx")

order_np = np.array(order_df)
supply_np = np.array(supply_df)

supply_np = np.where(supply_np < order_np, supply_np, order_np)

supply_np = torch.from_numpy(supply_np)
order_np = torch.from_numpy(order_np)
# continuity_ratio = 0.61  # data是0.9，data2是0.7

order_np = order_np
supply_np = supply_np   # note 这个数据集不转

# for i in range(3, supply_np.shape[0]):
#     supply_np[i] = (supply_np[i - 3] + supply_np[i - 2] + supply_np[i - 1]) / 3 * continuity_ratio + supply_np[i] * (1 - continuity_ratio)
#
# for i in range(3, order_np.shape[0]):
#     order_np[i] = (order_np[i - 3] + order_np[i - 2] + order_np[i - 1]) / 3 * continuity_ratio + order_np[i] * (1 - continuity_ratio)

supply_np = supply_np.numpy()
order_np = order_np.numpy()

np.save("order.npy", order_np)
np.save("supply.npy", supply_np)