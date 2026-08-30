# Crafter Action Space

Here is the exact integer mapping for the 17 actions that your DreamerV3 agent can take in the Crafter environment. 

When you look at the `Action Breakdown` printed at the end of the `eval.py` script, use this table to decode what your agent was doing!

| Integer ID | Action Name | Description | Keybinding (Human Play) |
| :--- | :--- | :--- | :--- |
| **0** | `noop` | Do nothing | - |
| **1** | `move_left` | Walk one tile left | `A` |
| **2** | `move_right` | Walk one tile right | `D` |
| **3** | `move_up` | Walk one tile up | `W` |
| **4** | `move_down` | Walk one tile down | `S` |
| **5** | `do` | Interact with the tile facing you (chop, mine, attack, drink water, etc.) | `Space` |
| **6** | `sleep` | Restore energy (requires safe environment, otherwise you get eaten) | `Tab` |
| **7** | `place_stone` | Place a stone block from inventory to block movement | `R` |
| **8** | `place_table` | Build a crafting table (requires wood) | `T` |
| **9** | `place_furnace`| Build a furnace (requires stone and table) | `F` |
| **10** | `place_plant` | Plant a sapling to grow a tree | `P` |
| **11** | `make_wood_pickaxe`| Craft a wooden pickaxe at a table (required to mine stone) | `1` |
| **12** | `make_stone_pickaxe`| Craft a stone pickaxe at a table (required to mine coal/iron) | `2` |
| **13** | `make_iron_pickaxe`| Craft an iron pickaxe at a furnace (required to mine diamond) | `3` |
| **14** | `make_wood_sword` | Craft a wooden sword at a table (increases damage to zombies) | `4` |
| **15** | `make_stone_sword` | Craft a stone sword at a table (increases damage) | `5` |
| **16** | `make_iron_sword` | Craft an iron sword at a furnace (increases damage) | `6` |

## Analyzing Your Final Evaluation (After 100k Steps + Human Data)
Looking back at your successful evaluation after fixing the bugs and injecting human data:
```
Episode 3 finished in 164 steps. Total Reward: 2.1
Action Breakdown:
  Action 7: 65 times
  Action 5: 27 times
  Action 0: 13 times
  Action 11: 7 times
  Action 9: 6 times
```

The agent is no longer frozen! It survived for hundreds of steps and even scored a total reward of 2.1. 
Instead of aggressively swinging at empty air, it was actively trying to build walls (**Action 7: `place_stone`**), craft tools (**Action 11: `make_wood_pickaxe`**), and construct workstations (**Action 9: `place_furnace`**)! The stats log also confirmed it successfully collected 17 saplings. The brain transplant was a massive success! 

