/* vibe.h: constants shared by the host programs (the library, the mel crops, the encoder's input). */
#ifndef VIBE_H
#define VIBE_H

/* the library root: $VIBE_LIBRARY, else the first line of ~/.config/phivibe/library, else ~/Music */
/* top folders left out (talk, not songs): -x on the command line, else one per line in ~/.config/phivibe/exclude */

#define MEL_SR 22050
#define MEL_NFFT 2048
#define MEL_HOP 1024
#define MEL_BANDS 64
#define MEL_FMIN 30.0
#define MEL_FRAMES 128                 /* one crop: 128 frames = 5.99 s */
#define MEL_CROPS 32                   /* at most, per song */
#define MEL_CROP_LEN ((MEL_FRAMES - 1) * MEL_HOP + MEL_NFFT)
#define MEL_CROP_BYTES (MEL_FRAMES * MEL_BANDS)
#define MEL_REC (MEL_CROPS * MEL_CROP_BYTES)
#define MEL_DB_MIN (-100.0)            /* u8 0 */
#define MEL_DB_MAX 10.0                /* u8 255 */

#endif
