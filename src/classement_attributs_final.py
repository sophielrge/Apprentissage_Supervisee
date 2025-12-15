feature_importances = {
    "WKHP": 0.0957,
    "AGEP": 0.0694,
    "4_Bachelors": 0.0355,
"5_Masters_to_Professional": 0.0351,
"3_HighSchool_to_Associates": 0.0090,
"SEX_2.0": 0.0083,
"RELP_2.0": 0.0079,
"OCCP_3255.0": 0.0074,
"RELP_17.0": 0.0072,
"6_Doctorate": 0.0063,
"OCCP_3602.0": 0.0060,
"OCCP_4720.0": 0.0057,
"MAR_5.0": 0.0048,
"COW_6.0": 0.0048,
"POBP_303.0": 0.0044,
"OCCP_6050.0": 0.0031,
"OCCP_4020.0": 0.0030,
"OCCP_4220.0": 0.0025,
"OCCP_4230.0": 0.0023,
"OCCP_1021.0": 0.0023,
"OCCP_2545.0": 0.0020,
"OCCP_4600.0": 0.0019,
"COW_3.0": 0.0018,
"OCCP_9620.0": 0.0018,
"OCCP_4110.0": 0.0017,
"OCCP_9645.0": 0.0017,
"RELP_10.0": 0.0015,
"OCCP_9640.0": 0.0015,
"RAC1P_6.0": 0.0015,
"OCCP_4030.0": 0.0014,
"OCCP_5400.0": 0.0014,
"OCCP_4760.0": 0.0014,
"OCCP_440.0": 0.0014,
"RELP_16.0": 0.0013,
"RELP_9.0": 0.0013,
"OCCP_4251.0": 0.0012,
"RELP_5.0": 0.0012,
"RAC1P_8.0": 0.0012,
"COW_5.0": 0.0012,
"RELP_12.0": 0.0011,
"OCCP_9142.0": 0.0011,
"OCCP_9130.0": 0.0010,
"OCCP_3930.0": 0.0010,
"OCCP_5740.0": 0.0009,
"OCCP_5610.0": 0.0009,
"OCCP_1360.0": 0.0008,
"OCCP_3603.0": 0.0008,
"OCCP_2300.0": 0.0008,
"COW_4.0": 0.0007,
"OCCP_3310.0": 0.0007,
"1_NoSchool_to_9th": 0.0007,
"POBP_6.0": 0.0007,
"OCCP_5240.0": 0.0007,
"OCCP_710.0": 0.0006,
"OCCP_8990.0": 0.0006,
"COW_7.0": 0.0006,
"OCCP_3870.0": 0.0006,
"OCCP_7750.0": 0.0006,
"RELP_7.0": 0.0006,
"OCCP_1530.0": 0.0006,
"RELP_4.0": 0.0006,
"POBP_312.0": 0.0006,
"OCCP_4700.0": 0.0005,
"OCCP_9600.0": 0.0005,
"OCCP_52.0": 0.0005,
"OCCP_4000.0": 0.0005,
"OCCP_705.0": 0.0005,
"OCCP_3090.0": 0.0005,
"OCCP_8320.0": 0.0005,
"OCCP_405.0": 0.0005,
"OCCP_3323.0": 0.0005,
"OCCP_350.0": 0.0005,
"RELP_13.0": 0.0004,
"OCCP_5860.0": 0.0004,
"RELP_15.0": 0.0004,
"OCCP_6355.0": 0.0004,
"MAR_3.0": 0.0004,
"OCCP_2205.0": 0.0004,
"OCCP_2310.0": 0.0004,
"OCCP_1010.0": 0.0004,
"OCCP_1108.0": 0.0004,
"OCCP_220.0": 0.0004,
"OCCP_120.0": 0.0004,
"OCCP_4140.0": 0.0004,
"OCCP_7800.0": 0.0004,
"OCCP_110.0": 0.0004,
"OCCP_2100.0": 0.0003
}


import pandas as pd
import matplotlib.pyplot as plt

df_importance = pd.DataFrame(
    feature_importances.items(),
    columns=["Feature", "Importance"]
)

df_importance = df_importance.sort_values(
    by="Importance",
    ascending=False
)

top_n = 15
df_top = df_importance.head(top_n)

plt.figure(figsize=(10, 6))
plt.barh(
    df_top["Feature"][::-1],
    df_top["Importance"][::-1]
)
plt.xlabel("Importance")
plt.title("Top 15 – Feature Importance")
plt.tight_layout()
plt.show()
