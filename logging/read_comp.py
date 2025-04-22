import pandas as pd
import numpy as np

df = pd.read_csv("/work/nvme/bdnb/atekkey/sam2/_results/comp2.csv") 

# print("\nALL FRAMES (USING SEPARATE-Prop for each video)")
# print("Backward stronger ", len(df[df["bp_gt"] > df["fp_gt"]]))
# print("Forward stronger ", len(df[df["bp_gt"] < df["fp_gt"]]))
# print("Equal or no GT present", len(df[df["bp_gt"] == df["fp_gt"]]))
# print("Total IOU Change:", (df["bp_gt"] - df["fp_gt"]).sum() )
# print("Count:",len(df))
# print("\n")

df1 = df.copy()

df_adj = df.copy()

print(len(df1[df1["gt_exists"] ==0 ]))

df_adj.loc[(df["bp_exists"] == 0) & (df["gt_exists"] == 0), "bp_gt"] = 1
df_adj.loc[(df["fp_exists"] == 0) & (df["gt_exists"] == 0), "fp_gt"] = 1


## GT Exists
# df1 = df1[df1["gt_exists"] > 0]
# fme = df1[df1["fp_exists"] > 0]
# fmne = df1[df1["fp_exists"] == 0]
# len_gte = len(df1)

# # fme_bmne = fme[fme["bp_exists"] <= 0]
# # print("FME vs BMNE:", 100* len(fme_bmne[fme_bmne["fp_gt"] > fme_bmne["bp_gt"]]) / len_gte, "%")



# fme_bme = fme[fme["bp_exists"] > 0]
# # fme_bme = fme_bme[fme_bme["inter"] < 0.1]
# # print(len(fme_bme))
# print("FME vs BME:", 100* len(fme_bme[fme_bme["fp_gt"] < fme_bme["bp_gt"]]) / len(fme_bme), "%")

# fme_bmne = fme[fme["bp_exists"] == 0]
# print("FME vs BMNE:", 100* len(fme_bmne[fme_bmne["fp_gt"] > fme_bmne["bp_gt"]]) / len(fme_bmne), "%")

# fmne_bmne = fmne[fmne["bp_exists"] <= 0]
# print("FMNE vs BMNE:", 100* len(fmne_bmne[fmne_bmne["fp_gt"] > fmne_bmne["bp_gt"]]) / len_gte, "%")
# fmne_bme = fmne[fmne["bp_exists"] > 0]
# print("FMNE vs BME:", 100* len(fmne_bme[fmne_bme["fp_gt"] < fmne_bme["bp_gt"]]) / len_gte, "%")

## GT DNE
df1 = df_adj.copy()
df1 = df1[df1["gt_exists"] <= 0]
fme = df1[df1["fp_exists"] <= 0]
fmne = df1[df1["fp_exists"] == 0]
len_gtne = len(df1)

fme_bme = (fme[fme["bp_exists"] > 0])
print(len(fme_bme[fme_bme["fp_gt"] > fme_bme["bp_gt"]]) *100 / len(fme_bme), "%")




print("ALL FRAMES")
df1 = df_adj.copy()
# df1 = df1[df1["fp_exists"] <= 0]
df1 = df1[df1["fp_exists"] == 0]
print(len(df1))
print(df1["fp_gt"].sum())
# print("Backward stronger ", len(df1[df1["bp_gt"] > df1["fp_gt"]]))
# print("Forward stronger ", len(df1[df1["bp_gt"] < df1["fp_gt"]]))
# print("Equal or no change", len(df1[df1["bp_gt"] == df1["fp_gt"]]))
# print("Total IOU Change:", (df1["bp_gt"] - df1["fp_gt"]).sum())
# print("Count:",len(df1))
# print("\n")

# df1 = df_adj.copy()
# fme = df1[df1["fp_exists"] == 0]
# print(fme["bp_gt"].sum())
# print(fme["fp_gt"].sum())

# df1 = df.copy()
# print(df1["bp_gt"].sum())
# print(df1["fp_gt"].sum())