/* irys over a C ABI (M1 item 7): render into an IOSurface with no copy.
 * Every function returns 0 / non-NULL on success; irys_last_error() has the
 * message otherwise. Handles are not thread-safe. */
#ifndef IRYS_H
#define IRYS_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct IrysContext IrysContext;
typedef struct IrysScene IrysScene;

const char *irys_last_error(void);

/* The process's GPU; NULL without one or without hardware ray queries. */
IrysContext *irys_context_create(void);
void irys_context_destroy(IrysContext *c);

/* A Blender export directory and / or a 3DGS .ply (either may be NULL; with
 * neither, a demo scene of four spheres), rendered at width x height. */
IrysScene *irys_scene_load(const IrysContext *c, const char *export_dir, const char *splat_ply,
                           uint32_t width, uint32_t height);
void irys_scene_destroy(IrysScene *s);

/* OpenCV pinhole intrinsics (pixels) and the camera -> world pose: rot is
 * row-major 3x3 whose columns are the optical axes (x right, y down, z
 * forward) in the world; pos is the camera centre. */
int irys_scene_set_camera(IrysScene *s, double fx, double fy, double cx, double cy,
                          const double rot[9], const double pos[3]);

/* Render spp samples into an IOSurfaceRef ('RGBA', 4 bytes per element, the
 * scene's size). Written on the GPU, nothing read back; done on return. */
int irys_scene_render_iosurface(IrysScene *s, const void *iosurface, uint32_t spp);
/* The same by IOSurfaceGetID() (looked up in this process). */
int irys_scene_render_iosurface_id(IrysScene *s, uint32_t iosurface_id, uint32_t spp);

/* The readback path, for comparison: sRGB8 RGBA, rows packed. */
int irys_scene_render_rgba8(IrysScene *s, uint8_t *out, uint32_t spp);

#ifdef __cplusplus
}
#endif
#endif
