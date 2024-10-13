# I used this script to pad the .bk2 files between Stage and Boss Chamber
# (CutMan-boss.state starts exactly 400 frames after finishing the stage state)
# and after killing the boss for that juicy ending.

with open("pad_before_boss.txt", "w") as f:
    for _ in range(400):
        f.write("|..|........|\n")
with open("pad_after_boss.txt", "w") as f:
    for _ in range(60 * 5):
        f.write("|..|.R......|\n")
    for _ in range(60 * 25):
        f.write("|..|........|\n")
