# Sliding Window Adversarial Attack - Simple Explanation

## The Problem with Your Old Approach

Imagine you have a video with 200 frames showing a car driving. Your old approach treated this like **one big movie**:

```
Old Approach:
┌────────────────────────────────────────────────────────────┐
│ [Frame 0][Frame 1][Frame 2]...[Frame 198][Frame 199]      │
│                    ONE SEQUENCE                             │
└────────────────────────────────────────────────────────────┘
                            ↓
              Process through SDNN model
                            ↓
            Get ONE target (average of 200 predictions)
                            ↓
            Perturb ALL 200 frames together
```

**Problem:** You're attacking the model's understanding of the ENTIRE 200-frame sequence as a whole, not individual temporal contexts.

---

## The New Sliding Window Approach

Instead, think of it like breaking the video into **many overlapping short clips**, and attacking each clip separately:

```
New Approach (Window Size = 5):

Window 0: [0][1][2][3][4]                                      ← Attack this independently
Window 1:    [1][2][3][4][5]                                   ← Attack this independently
Window 2:       [2][3][4][5][6]                                ← Attack this independently
Window 3:          [3][4][5][6][7]                             ← Attack this independently
...
Window 195:                         ...[195][196][197][198][199] ← Attack this independently

Total: 196 separate attacks!
```

---

## Key Insight: Frames Get Attacked Multiple Times

Look at **Frame 2**:
- It appears in Window 0: `[0][1][2][3][4]`
- It appears in Window 1: `[1][2][3][4][5]`
- It appears in Window 2: `[2][3][4][5][6]`
- It appears in Window 3: `[3][4][5][6][7]`
- It appears in Window 4: `[4][5][6][7][8]`

That's **5 different attacks** on Frame 2, each in a different temporal context!

### Why Is This Better?

Each time Frame 2 is attacked, the model sees it in a **different context**:

1. **Window 0** `[0,1,2,3,4]`: Frame 2 is the "middle" frame
   - Model prediction based on frames before and after it
   
2. **Window 1** `[1,2,3,4,5]`: Frame 2 is "early-middle"
   - Different neighboring frames, different prediction
   
3. **Window 2** `[2,3,4,5,6]`: Frame 2 is the "first" frame
   - No prior frames in this window, different context

Each attack generates a **different perturbation** for Frame 2. We average these 5 perturbations to create a **more robust adversarial perturbation**.

---

## Visual Example: How Frame 50 Gets Perturbed

```
Original Frame 50: 🚗

Window 45: [45,46,47,48,49] ... [50] ← Attack 1 → Perturbation A
Window 46: [46,47,48,49,50] ... [51] ← Attack 2 → Perturbation B  
Window 47: [47,48,49,50,51] ... [52] ← Attack 3 → Perturbation C
Window 48: [48,49,50,51,52] ... [53] ← Attack 4 → Perturbation D
Window 49: [49,50,51,52,53] ... [54] ← Attack 5 → Perturbation E

Final Perturbation for Frame 50 = Average(A, B, C, D, E)

Adversarial Frame 50: 🚗💥 (with averaged perturbation)
```

---

## Step-by-Step: What the Script Does

### Step 1: Load All Images
```python
Load: [Frame 0, Frame 1, Frame 2, ..., Frame 199]
Total: 200 frames
```

### Step 2: Create Windows
```python
Window 0:   frames [0, 1, 2, 3, 4]
Window 1:   frames [1, 2, 3, 4, 5]
Window 2:   frames [2, 3, 4, 5, 6]
...
Window 195: frames [195, 196, 197, 198, 199]

Total: 196 windows
```

### Step 3: Attack Each Window Independently

For **Window 0** `[0,1,2,3,4]`:
1. Feed these 5 frames to SDNN → Get prediction
2. Calculate target value (average of predictions)
3. Run adversarial attack (FGSM/PGD/MIFGSM) on these 5 frames
4. Get perturbation for frames 0,1,2,3,4
5. **Save** this perturbation

For **Window 1** `[1,2,3,4,5]`:
1. Feed these 5 frames to SDNN → Get prediction (different from Window 0!)
2. Calculate target value for THIS window
3. Run adversarial attack on these 5 frames
4. Get perturbation for frames 1,2,3,4,5
5. **Save** this perturbation

... Repeat for all 196 windows ...

### Step 4: Combine Overlapping Perturbations

Now Frame 2 has perturbations from 5 different windows. We **average** them:

```python
Frame 2's final perturbation = (
    Perturbation from Window 0 +
    Perturbation from Window 1 +
    Perturbation from Window 2 +
    Perturbation from Window 3 +
    Perturbation from Window 4
) / 5
```

### Step 5: Apply to Original Images

```python
Adversarial Frame 2 = Original Frame 2 + Final Perturbation
```

---

## How Many Times Is Each Frame Attacked?

With 200 frames and window size 5:

```
Frame 0:   Appears in 1 window  → Attacked 1 time
Frame 1:   Appears in 2 windows → Attacked 2 times
Frame 2:   Appears in 3 windows → Attacked 3 times
Frame 3:   Appears in 4 windows → Attacked 4 times
Frame 4:   Appears in 5 windows → Attacked 5 times
Frame 5:   Appears in 5 windows → Attacked 5 times
...
Frame 194: Appears in 5 windows → Attacked 5 times
Frame 195: Appears in 5 windows → Attacked 5 times
Frame 196: Appears in 4 windows → Attacked 4 times
Frame 197: Appears in 3 windows → Attacked 3 times
Frame 198: Appears in 2 windows → Attacked 2 times
Frame 199: Appears in 1 window  → Attacked 1 time
```

**Middle frames (5-194)** are each attacked **5 times** from different temporal contexts!

---

## Why Professors Recommend This

### Professor's Quote:
> "You shouldn't perturb all 200 at once. You should treat them as... Whatever, 200 divided by 5, like 40 different samples... Each of those 5 frames could be perturbed differently."

### Benefits:

1. **✅ More thorough testing**
   - Each temporal context (5-frame window) is attacked independently
   - Tests model's robustness across different temporal segments

2. **✅ More realistic**
   - In real-world, model processes short temporal sequences
   - This mimics how model would actually be used

3. **✅ Stronger attacks**
   - Frames attacked from multiple perspectives
   - Averaged perturbations are more effective

4. **✅ Scalable to full test set**
   - Not limited to one 200-frame sequence
   - Can run on entire test dataset

---

## Usage Example

```bash
# Old approach (attacks all 200 frames as one sequence)
python generate_adversarial_torchattacks.py --attack PGD --eps 0.03 --alpha 0.007 --steps 10

# New approach (attacks overlapping 5-frame windows)
python generate_adversarial_sliding_window.py --attack PGD --eps 0.03 --alpha 0.007 --steps 10 --window_size 5
```

---

## Expected Output

```
==================================================================
SLIDING WINDOW ADVERSARIAL ATTACK - PGD
==================================================================
Total frames: 200
Window size: 5
Number of windows: 196
==================================================================

Processing Window 1/196: frames [0 to 4] | Target: -0.248081 | Pert norm: 0.015234
Processing Window 2/196: frames [1 to 5] | Target: -0.251203 | Pert norm: 0.014891
Processing Window 3/196: frames [2 to 6] | Target: -0.249456 | Pert norm: 0.015102
...
Processing Window 196/196: frames [195 to 199] | Target: -0.223456 | Pert norm: 0.014567

==================================================================
Averaging overlapping perturbations...
==================================================================

Perturbation count per frame:
  Frames 0-3: attacked [1.0, 2.0, 3.0, 4.0] times
  Frames 4-195: attacked 5.0 times each
  Frames 196-199: attacked [4.0, 3.0, 2.0, 1.0] times
```

---

## Comparison Summary

| Old Approach | New Sliding Window Approach |
|-------------|----------------------------|
| 1 attack (entire sequence) | 196 attacks (overlapping windows) |
| Each frame attacked once | Middle frames attacked 5 times |
| One global target | 196 different targets |
| Fast (single pass) | Slower (196 passes) |
| Less thorough | More thorough |

---

## Bottom Line

**Old:** "Attack the movie as a whole"
**New:** "Attack many short clips separately, then combine"

The new approach is what your professors recommended because it's **more thorough** and **more realistic** for testing adversarial robustness of temporal models like SDNNs.
