"""C3VD colonoscope sensor.

Scaramuzza model from Bobrow et al., MedIA 2023, Eq. 3–4.
Pixel (u, v) is in the image frame (origin at the top-left, v downward).
The ray in that frame is (u', v', f(rho)), which is OpenCV camera space
(X right, Y down, Z forward). The sensor `to_world` is the Mitsuba camera
(X left, Y up, Z forward), so the local direction flips X and Y.
"""

import drjit as dr
import mitsuba as mi

mi.set_variant("cuda_rgb")


# cfhq190l_omnidirectional_values.txt / paper Table 3.
# The file columns e, f, g are the stretch matrix [[e, f], [g, 1]].
DEFAULTS = {
    "width": 1350.0,
    "height": 1080.0,
    "cx": 679.544839263292,
    "cy": 543.975887548343,
    "a0": 769.243600037458,
    "a2": -0.000812770624150226,
    "a3": 6.25674244578925e-07,
    "a4": -1.19662182144280e-09,
    "e": 0.999986882249990,
    "f": 0.00288273829525059,
    "g": -0.00296316513429569,
    "near_clip": 0.1,
    "far_clip": 200.0,
}


def _float_prop(props, name):
    if name in props:
        return float(props[name])
    return float(DEFAULTS[name])


def sensor_prefix(params) -> str:
    """Parameter prefix of the scene's sensor, e.g. ``C3VDCamera``."""
    suffix = ".film.size"
    for key in params.keys():
        if key.endswith(suffix):
            return key[: -len(suffix)]
    raise KeyError("scene has no sensor film")


class C3VDCamera(mi.Sensor):
    def __init__(self, props):
        super().__init__(props)
        self.m_needs_sample_3 = False
        self.cx = _float_prop(props, "cx")
        self.cy = _float_prop(props, "cy")
        self.a0 = _float_prop(props, "a0")
        self.a2 = _float_prop(props, "a2")
        self.a3 = _float_prop(props, "a3")
        self.a4 = _float_prop(props, "a4")
        self.e = _float_prop(props, "e")
        self.f = _float_prop(props, "f")
        self.g = _float_prop(props, "g")
        self.near_clip = _float_prop(props, "near_clip")
        self.far_clip = _float_prop(props, "far_clip")
        self._to_world = mi.Transform4f(self.world_transform())
        self._cache_film()

    def _cache_film(self):
        film = self.film()
        crop = film.crop_size()
        offset = film.crop_offset()
        self._width = float(crop[0])
        self._height = float(crop[1])
        self._off_x = float(offset[0])
        self._off_y = float(offset[1])

    def traverse(self, callback):
        super().traverse(callback)
        callback.put_parameter(
            "to_world", self._to_world, mi.ParamFlags.NonDifferentiable
        )

    def parameters_changed(self, keys):
        super().parameters_changed(keys)
        self._cache_film()

    def sample_ray(self, time, wavelength_sample, position_sample, aperture_sample, active=True):
        del aperture_sample
        wavelengths, weight = self.sample_wavelengths(
            dr.zeros(mi.SurfaceInteraction3f), wavelength_sample, active
        )

        # Continuous pixel coordinate: left/top edge is 0, right/bottom edge is width/height.
        u = self._off_x + position_sample.x * self._width
        v = self._off_y + position_sample.y * self._height
        px = u - self.cx
        py = v - self.cy

        det = self.e - self.f * self.g
        u_p = (px - self.f * py) / det
        v_p = (-self.g * px + self.e * py) / det
        rho = dr.sqrt(u_p * u_p + v_p * v_p)
        rho2 = rho * rho
        poly = self.a0 + self.a2 * rho2 + self.a3 * rho2 * rho + self.a4 * rho2 * rho2

        # OpenCV (x right, y down, z forward) -> Mitsuba (x left, y up, z forward).
        direction = dr.normalize(mi.Vector3f(-u_p, -v_p, poly))
        direction = self._to_world @ direction
        origin = self._to_world.translation() + direction * self.near_clip

        ray = mi.Ray3f(origin, direction)
        ray.time = time
        ray.wavelengths = wavelengths
        ray.maxt = mi.Float(self.far_clip - self.near_clip)
        return ray, weight

    def sample_direction(self, it, sample, active=True):
        del it, sample, active
        return dr.zeros(mi.DirectionSample3f), dr.zeros(mi.Color3f)

    def pdf_direction(self, it, ds, active=True):
        del it, ds, active
        return mi.Float(0.0)

    def eval_direction(self, it, ds, active=True):
        del it, ds, active
        return dr.zeros(mi.Color3f)

    def sample_position(self, time, sample, active=True):
        del time, sample, active
        return dr.zeros(mi.PositionSample3f), mi.Float(0.0)

    def pdf_position(self, ps, active=True):
        del ps, active
        return mi.Float(0.0)

    def eval(self, si, active=True):
        del si, active
        return dr.zeros(mi.Color3f)

    def bbox(self):
        origin = self.world_transform().translation()
        point = mi.ScalarPoint3f(float(origin[0]), float(origin[1]), float(origin[2]))
        return mi.ScalarBoundingBox3f(point, point)

    def to_string(self):
        return (
            f"C3VDCamera[cx={self.cx}, cy={self.cy}, a0={self.a0}, "
            f"near={self.near_clip}, far={self.far_clip}]"
        )


def register_c3vd_sensor():
    """Register the sensor for the active variant. Safe to call more than once."""
    mi.set_variant("cuda_rgb")
    try:
        mi.register_sensor("c3vd", C3VDCamera)
    except Exception as exc:
        if "already" not in str(exc).lower():
            raise


register_c3vd_sensor()
