#ifdef _MSC_VER
#ifndef HAVE_SSIZE_T
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif
#else
#include <sys/types.h>
#endif

#ifndef RESTRICT
#if defined(_MSC_VER)
#define RESTRICT __restrict
#elif defined(__GNUC__)
#define RESTRICT __restrict__
#elif (__STDC_VERSION__ >= 199901L)
#define RESTRICT restrict
#else
#define RESTRICT
#endif
#endif

#ifndef ALIGNED
#if defined(_MSC_VER)
#define ALIGNED(t, n) __declspec(align(n)) t
#elif defined(__GNUC__)
#define ALIGNED(t, n) t __attribute__((aligned(8)))
#else
#define ALIGNED(t, n) t
#endif
#endif

typedef ALIGNED(double, 8) dbl_algn;
typedef dbl_algn * RESTRICT dbl_ptr_ar; // aligned, restricted
typedef const dbl_algn * RESTRICT dbl_ptr_car; // const, aligned, restricted

#ifdef __cplusplus
extern "C" {
#endif

/**
 * HoG filtering.
 *   image is in pixels of w and h, must be C order
 *   out is where data is saved, it is n pixels long
 *   returns:
 *     -1 if out is not long enough
 *     -2 if temporary memory can't be allocated
 *     otherwise it returns the number of values written to out
 *
 * This essentially calls HoG_init then Hog_run.
 */
ssize_t HoG(dbl_ptr_car pixels, const ssize_t w, const ssize_t h, dbl_ptr_ar out, const ssize_t n);

/**
 * Checks and initilization for HoG filtering.
 *   the image size is given in w/h
 *   the total room needed in the output is given in n (in number of pixels)
 *   returns the number of pixels needed in the temporary buffer
 */
ssize_t HoG_init(const ssize_t w, const ssize_t h, ssize_t *n);

/**
 * The core code for the HoG filtering.
 *   image is in pixels of w and h, must be C order
 *   out is where data is saved, it must be at least n pixels long (the n from HoG_init)
 *   H is a temporary buffer with the number of elements returned by HoG_init
 */
void HoG_run(dbl_ptr_car pixels, const ssize_t w, const ssize_t h, dbl_ptr_ar out, dbl_ptr_ar H);


/** The original, nearly unmodified, HoG code. */
void HoG_orig(double *pixels, double *params, int *img_size, double *dth_des, unsigned int grayscale);

#ifdef __cplusplus
}
#endif
