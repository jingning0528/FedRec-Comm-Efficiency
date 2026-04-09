import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 14, 'font.family': 'serif'})

# Epochs
epochs = list(range(50))

# Random method
random_hit = [
    0.2589, 0.283, 0.3126, 0.3292, 0.3535, 0.3831, 0.4093,
    0.4284, 0.4487, 0.4641, 0.4711, 0.472, 0.4744,
    0.4756, 0.4789, 0.4782, 0.4863, 0.4862,
    0.4912, 0.4963, 0.5022, 0.5026, 0.503, 0.517,
    0.5158, 0.5217, 0.5259, 0.5338, 0.5312, 0.5341,
    0.5391, 0.5502, 0.551, 0.5678, 0.5674, 0.5693,
    0.5665, 0.564, 0.5586, 0.5603, 0.5594, 0.5654,
    0.5664, 0.5652, 0.5623, 0.565, 0.5729, 0.5759,
    0.5786, 0.5803
]

random_ndcg = [
    0.0007, 0.0055, 0.019, 0.039, 0.0724, 0.1001, 0.1336,
    0.1648, 0.1957, 0.2056, 0.2317, 0.2521, 0.2635,
    0.2766, 0.2917, 0.305, 0.3204, 0.3332,
    0.3448, 0.3555, 0.3649, 0.3763, 0.3855, 0.3965,
    0.4064, 0.4153, 0.4228, 0.4279, 0.4342, 0.4408,
    0.4467, 0.4494, 0.4569, 0.4581, 0.464, 0.467,
    0.4718, 0.474, 0.4792, 0.481, 0.4837, 0.488,
    0.4925, 0.4959, 0.4987, 0.5033, 0.5038, 0.5038,
    0.507, 0.5078
]

# Item LLM Init method (previously "LLM")
item_llm_hit = [
    0.2986, 0.3171, 0.325, 0.3564, 0.3861, 0.4059, 0.4292,
    0.465, 0.4848, 0.4935, 0.4962, 0.4939, 0.5047,
    0.5011, 0.5027, 0.5049, 0.5086, 0.5162,
    0.5178, 0.5336, 0.5432, 0.5484, 0.5532, 0.5556,
    0.5523, 0.5546, 0.5661, 0.5686, 0.5792, 0.5785,
    0.5851, 0.5911, 0.5889, 0.5878, 0.5886, 0.5887,
    0.5835, 0.5841, 0.5898, 0.5936, 0.5871, 0.5873,
    0.5903, 0.5943, 0.5936, 0.5975, 0.5975, 0.5988,
    0.5977, 0.6015
]

item_llm_ndcg = [
    0.0017, 0.0141, 0.035, 0.0677, 0.0995, 0.1436, 0.1799,
    0.2057, 0.234, 0.2585, 0.269, 0.2832, 0.3018,
    0.3176, 0.3328, 0.3497, 0.3637, 0.376,
    0.389, 0.4034, 0.4159, 0.429, 0.4358, 0.4418,
    0.4476, 0.4587, 0.4644, 0.4707, 0.4713, 0.4787,
    0.4818, 0.4869, 0.4942, 0.497, 0.5002, 0.5062,
    0.5067, 0.5142, 0.5115, 0.5134, 0.5157, 0.5199,
    0.521, 0.5229, 0.5229, 0.5261, 0.5298, 0.5288,
    0.5313, 0.534
]

# User LLM Init method (item LLM init + user init from item avg)
user_llm_hit = [
    0.2524, 0.3087, 0.3519, 0.3872, 0.4218, 0.4328, 0.4649,
    0.4712, 0.4831, 0.4828, 0.4835, 0.4822, 0.4783,
    0.5031, 0.5016, 0.5026, 0.5154, 0.5195,
    0.5185, 0.5177, 0.5258, 0.5396, 0.5498, 0.5509,
    0.5634, 0.5601, 0.5679, 0.5703, 0.5717, 0.57,
    0.568, 0.5697, 0.5693, 0.5763, 0.5767, 0.5724,
    0.5835, 0.5787, 0.5808, 0.5819, 0.5875, 0.5883,
    0.5897, 0.5898, 0.5897, 0.5961, 0.601, 0.5947,
    0.6048, 0.6058
]

user_llm_ndcg = [
    0.0033, 0.0157, 0.04, 0.0745, 0.1053, 0.1419, 0.1729,
    0.1998, 0.22, 0.2524, 0.2639, 0.2762, 0.2936,
    0.3101, 0.3276, 0.3405, 0.3566, 0.3684,
    0.3797, 0.3914, 0.4039, 0.4155, 0.4239, 0.4349,
    0.4432, 0.4485, 0.4564, 0.4591, 0.4647, 0.4697,
    0.4761, 0.4789, 0.4845, 0.4873, 0.4889, 0.494,
    0.4995, 0.5013, 0.5037, 0.5086, 0.5103, 0.5105,
    0.5117, 0.5137, 0.5154, 0.5171, 0.5207, 0.521,
    0.5225, 0.5218
]

# -------- Figure: Hit@10 and NDCG@10 side by side --------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Hit@10
ax1.plot(epochs, random_hit, 'o-', color='#2196F3', linewidth=2, markersize=3, label='Random Init')
ax1.plot(epochs, item_llm_hit, 's-', color='#F44336', linewidth=2, markersize=3, label='Item LLM Init (E5-base-v2)')
ax1.plot(epochs, user_llm_hit, 'D-', color='#4CAF50', linewidth=2, markersize=3, label='User LLM Init (Item Avg)')
ax1.set_xlabel('Communication Rounds', fontsize=14)
ax1.set_ylabel('HR@10', fontsize=14)
ax1.set_title('(a) HR@10 vs Communication Rounds', fontsize=15)
ax1.legend(fontsize=11, loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-0.5, 49.5)
ax1.set_ylim(0.2, 0.65)

# NDCG@10
ax2.plot(epochs, random_ndcg, 'o-', color='#2196F3', linewidth=2, markersize=3, label='Random Init')
ax2.plot(epochs, item_llm_ndcg, 's-', color='#F44336', linewidth=2, markersize=3, label='Item LLM Init (E5-base-v2)')
ax2.plot(epochs, user_llm_ndcg, 'D-', color='#4CAF50', linewidth=2, markersize=3, label='User LLM Init (Item Avg)')
ax2.set_xlabel('Communication Rounds', fontsize=14)
ax2.set_ylabel('NDCG@10', fontsize=14)
ax2.set_title('(b) NDCG@10 vs Communication Rounds', fontsize=15)
ax2.legend(fontsize=11, loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-0.5, 49.5)
ax2.set_ylim(0.0, 0.6)

plt.tight_layout()
plt.savefig('llm_vs_random_init.pdf', dpi=300, bbox_inches='tight')
plt.savefig('llm_vs_random_init.png', dpi=300, bbox_inches='tight')
plt.show()

# -------- Print convergence analysis --------
print("\n=== Convergence Analysis ===")
targets_hr = [0.40, 0.45, 0.50, 0.55, 0.58]
targets_ndcg = [0.20, 0.30, 0.40, 0.45, 0.50]

print("\nHR@10 Target | Random (round) | Item LLM (round) | User LLM (round) | Saved (Item) | Saved (User)")
print("-" * 95)
for target in targets_hr:
    r_round = next((i for i, v in enumerate(random_hit) if v >= target), None)
    il_round = next((i for i, v in enumerate(item_llm_hit) if v >= target), None)
    ul_round = next((i for i, v in enumerate(user_llm_hit) if v >= target), None)
    saved_item = f"{r_round - il_round}" if r_round is not None and il_round is not None else "N/A"
    saved_user = f"{r_round - ul_round}" if r_round is not None and ul_round is not None else "N/A"
    r_str = str(r_round) if r_round is not None else "N/A"
    il_str = str(il_round) if il_round is not None else "N/A"
    ul_str = str(ul_round) if ul_round is not None else "N/A"
    print(f"  {target:.2f}       |      {r_str:>3}       |       {il_str:>3}         |       {ul_str:>3}         |     {saved_item:>3}      |     {saved_user:>3}")

print("\nNDCG@10 Target | Random (round) | Item LLM (round) | User LLM (round) | Saved (Item) | Saved (User)")
print("-" * 95)
for target in targets_ndcg:
    r_round = next((i for i, v in enumerate(random_ndcg) if v >= target), None)
    il_round = next((i for i, v in enumerate(item_llm_ndcg) if v >= target), None)
    ul_round = next((i for i, v in enumerate(user_llm_ndcg) if v >= target), None)
    saved_item = f"{r_round - il_round}" if r_round is not None and il_round is not None else "N/A"
    saved_user = f"{r_round - ul_round}" if r_round is not None and ul_round is not None else "N/A"
    r_str = str(r_round) if r_round is not None else "N/A"
    il_str = str(il_round) if il_round is not None else "N/A"
    ul_str = str(ul_round) if ul_round is not None else "N/A"
    print(f"  {target:.2f}       |      {r_str:>3}       |       {il_str:>3}         |       {ul_str:>3}         |     {saved_item:>3}      |     {saved_user:>3}")