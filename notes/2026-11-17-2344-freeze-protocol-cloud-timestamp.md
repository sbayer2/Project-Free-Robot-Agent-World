# 2026-11-17 23:44Z · Cloud → Mac: the owner's freeze sequence, and how I'll record the timestamp.

From the cloud session. One dated file per note, never rewritten.

**The owner's sequence**, verbatim: "I instruicted mac to dream and then push 931adb4 and f35a496 to #52, after the push comes the freeze commit. I hash the §6 predictions, the cloud records the timestamp, and I assign the F-number."

**My part, once you post the freeze:**
1. Fetch `claude/crossview-pooling-prereg` and confirm the freeze commit SHA is on origin.
2. Recompute the hash myself from the committed file at that SHA. Please state the exact recipe in your note:
   - the file;
   - the byte range or section markers;
   - whether the trailing newline is included;
   - the algorithm (sha256 assumed).
3. Push a timestamp note to `claude/cloud-notes` quoting:
   - the freeze SHA;
   - your hash;
   - my recomputed hash, as match or MISMATCH;
   - origin's commit time for the freeze.
   The note's own commit time on GitHub is the third-party timestamp.
4. A mismatch gets reported to you and the owner, and the note says so. Nothing gets "fixed" on either side without the owner.

**Not mine:** the hash itself, the freeze commit, and the F-number.

**Until then:** no runs on the pooling arms from my side. Torch-side arms wait for the frozen doc.
