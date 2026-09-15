from build123d import Box, Cylinder, Pos

from articraft.sdk import (
    JointAxis,
    JointDOF,
    Material,
    RigidBodyAssembly,
    TestContext,
    TestReport,
)

# --- Overall dimensions (meters) ---
W = 1.20          # width  (X)
D = 0.45          # depth  (Y, front faces +Y)
H = 0.90          # height (Z)
FOOT_H = 0.10     # height of the four feet
CARC_Z0 = FOOT_H  # carcass bottom
PANEL = 0.022     # carcass panel / divider / rail thickness

WHITE = (0.92, 0.92, 0.90)
BRASS = Material(name="brass", density=8700.0, friction=(0.40, 0.35))
BRASS_COL = (0.74, 0.58, 0.26)
WOOD = Material.HARDWOOD

# --- Carcass interior grid ---
X_IN0 = -(W / 2 - PANEL)      # -0.578
X_IN1 = (W / 2 - PANEL)       #  0.578
Z_IN0 = CARC_Z0 + PANEL       # top of bottom panel
Z_IN1 = H - PANEL             # bottom of top panel

# 3 columns, 2 vertical dividers
COL_W = (X_IN1 - X_IN0 - 2 * PANEL) / 3.0
COL_CX = [X_IN0 + COL_W / 2.0 + i * (COL_W + PANEL) for i in range(3)]

# 3 rows, 2 horizontal rails
ROW_H = (Z_IN1 - Z_IN0 - 2 * PANEL) / 3.0
ROW_CZ = [Z_IN0 + ROW_H / 2.0 + i * (ROW_H + PANEL) for i in range(3)]

# --- Drawer box / front sizing ---
FRONT_TH = 0.018
FRONT_Y1 = D                  # 0.45 front face
FRONT_Y0 = D - FRONT_TH       # 0.432
FRONT_W = COL_W - 0.018
FRONT_H = ROW_H - 0.012
BOX_Y0 = 0.16                 # back of the drawer box
BOX_Y1 = FRONT_Y0 + 0.004    # box ends just behind the front panel face
BOX_W = COL_W - 0.030        # narrower than the front, tucks behind it
BOX_H = ROW_H - 0.020
BOX_WALL = 0.006
TRAVEL = 0.28               # prismatic travel of each drawer


def build_object_model() -> RigidBodyAssembly:
    model = RigidBodyAssembly("chest_of_drawers")

    carcass = model.rigid_body("carcass")

    # Two side panels (inner face at +/- (W/2 - PANEL))
    for sx, name in ((W / 2 - PANEL / 2.0, "side_right"), (-(W / 2 - PANEL / 2.0), "side_left")):
        carcass.add(
            Pos(X=sx, Y=D / 2.0, Z=(CARC_Z0 + H) / 2.0) * Box(PANEL, D, H - CARC_Z0),
            name=name,
            material=WOOD,
            color=WHITE,
        )
    # Top and bottom
    carcass.add(
        Pos(X=0.0, Y=D / 2.0, Z=H - PANEL / 2.0) * Box(W, D, PANEL),
        name="top",
        material=WOOD,
        color=WHITE,
    )
    carcass.add(
        Pos(X=0.0, Y=D / 2.0, Z=CARC_Z0 + PANEL / 2.0) * Box(W, D, PANEL),
        name="bottom",
        material=WOOD,
        color=WHITE,
    )
    # Back panel
    carcass.add(
        Pos(X=0.0, Y=PANEL / 2.0, Z=(CARC_Z0 + H) / 2.0) * Box(W, PANEL, H - CARC_Z0),
        name="back",
        material=WOOD,
        color=WHITE,
    )
    # Two vertical dividers
    for i, dx in enumerate([X_IN0 + COL_W + PANEL / 2.0, X_IN0 + 2 * COL_W + 3 * PANEL / 2.0]):
        carcass.add(
            Pos(X=dx, Y=D / 2.0, Z=(CARC_Z0 + H) / 2.0) * Box(PANEL, D, H - CARC_Z0),
            name=f"divider_{i}",
            material=WOOD,
            color=WHITE,
        )
    # Two horizontal rails (between the side panels, full depth)
    for i, dz in enumerate(RAIL_CZ()):
        carcass.add(
            Pos(X=0.0, Y=D / 2.0, Z=dz) * Box(W, D, PANEL),
            name=f"rail_{i}",
            material=WOOD,
            color=WHITE,
        )
    # Four feet
    for i, (fx, fy) in enumerate(
        [(-0.56, 0.41), (0.56, 0.41), (-0.56, 0.04), (0.56, 0.04)]
    ):
        carcass.add(
            Pos(X=fx, Y=fy, Z=FOOT_H / 2.0) * Box(0.035, 0.035, FOOT_H),
            name=f"foot_{i}",
            material=WOOD,
            color=WHITE,
        )

    # --- Drawers ---
    joint_names = []
    for r in range(3):
        for c in range(3):
            cx = COL_CX[c]
            cz = ROW_CZ[r]
            drawer = model.rigid_body(f"drawer_{r}_{c}")
            for shp, name, mat, col in _make_drawer_front(cx, cz):
                drawer.add(shp, name=name, material=mat, color=col)
            # Prismatic joint: the drawer slides straight out along +Y.
            pn = f"drawer_{r}_{c}_slide"
            joint_names.append(pn)
            model.joint(
                pn,
                carcass.at((cx, 0.30, cz)),
                drawer.at((cx, 0.30, cz)),
                dofs=(JointDOF(JointAxis.TRANS_Y, limits=(0.0, TRAVEL)),),
            )

    model.articulation("main", root=carcass, joints=joint_names)
    return model


def RAIL_CZ():
    return [Z_IN0 + ROW_H + PANEL / 2.0 + i * (ROW_H + PANEL) for i in range(2)]


def _make_drawer_front(cx: float, cz: float):
    """Return (shape, name, material, color) tuples for one drawer."""
    # Fluted front: a board with fine vertical grooves cut into the +Y face.
    front = Pos(X=cx, Y=(FRONT_Y0 + FRONT_Y1) / 2.0, Z=cz) * Box(
        FRONT_W, FRONT_TH, FRONT_H
    )
    n_grooves = 17
    pitch = FRONT_W / (n_grooves + 1)
    groove_cutters = []
    for i in range(n_grooves):
        gx = cx - FRONT_W / 2.0 + pitch * (i + 1)
        groove_cutters.append(
            Pos(X=gx, Y=FRONT_Y1 - 0.002, Z=cz) * Cylinder(radius=0.006, height=FRONT_H + 0.01)
        )
    cutter = groove_cutters[0]
    for g in groove_cutters[1:]:
        cutter = cutter.fuse(g)
    front = front.cut(cutter)

    # Drawer box behind the front (thin-walled, open at the front face).
    outer = Pos(X=cx, Y=(BOX_Y0 + BOX_Y1) / 2.0, Z=cz) * Box(
        BOX_W, BOX_Y1 - BOX_Y0, BOX_H
    )
    cavity_len = (BOX_Y1 - BOX_Y0) - BOX_WALL          # open at the front
    cavity_center_y = BOX_Y0 + BOX_WALL / 2.0 + cavity_len / 2.0
    inner = Pos(X=cx, Y=cavity_center_y, Z=cz) * Box(
        BOX_W - 2 * BOX_WALL, cavity_len, BOX_H - 2 * BOX_WALL
    )
    box = outer.cut(inner)

    # Brass bar handle standing proud at the centre of the front.
    handle = Pos(X=cx, Y=FRONT_Y1 + 0.006, Z=cz) * Box(0.006, 0.018, 0.062)
    handle_back = Pos(X=cx, Y=FRONT_Y1 - 0.004, Z=cz) * Box(0.014, 0.012, 0.070)

    # Brass corner brackets at the four corners of the front.
    brackets = []
    for sx in (-1, 1):
        for sz in (-1, 1):
            brackets.append(
                Pos(
                    X=cx + sx * (FRONT_W / 2.0 - 0.012),
                    Y=FRONT_Y1 - 0.006,
                    Z=cz + sz * (FRONT_H / 2.0 - 0.010),
                )
                * Box(0.016, 0.014, 0.016)
            )

    return [
        (front, "front", WOOD, WHITE),
        (box, "drawer_box", WOOD, WHITE),
        (handle, "handle", BRASS, BRASS_COL),
        (handle_back, "handle_back", BRASS, BRASS_COL),
    ] + [
        (b, f"bracket_{i}", BRASS, BRASS_COL) for i, b in enumerate(brackets)
    ]


object_model = build_object_model()


def run_tests() -> TestReport:
    ctx = TestContext(object_model)

    m = ctx.measure_geometry()
    ctx.record_metric("width_x", m.dimensions[0], unit="m")
    ctx.record_metric("depth_y", m.dimensions[1], unit="m")
    ctx.record_metric("height_z", m.dimensions[2], unit="m")
    ctx.expect_bounds(
        minimum=(-0.6, 0.0, 0.0),
        maximum=(0.6, 0.47, 0.9),
        tolerance=0.008,
    )

    for r in range(3):
        for c in range(3):
            pn = f"drawer_{r}_{c}_slide"
            # At rest the drawer sits in its opening without hitting the carcass.
            ctx.expect_no_collision(
                f"drawer_{r}_{c}", "carcass", name=f"rest_{r}_{c}"
            )
            # Pulled fully open it slides out and stays clear of the carcass.
            # Pulled fully open it slides out along +Y and stays clear.
            with ctx.pose({pn: TRAVEL}):
                ctx.expect_no_collision(
                    f"drawer_{r}_{c}", "carcass", name=f"open_{r}_{c}"
                )
                b = ctx.shape_world_bounds(f"drawer_{r}_{c}", "front")
                ctx.expect_metric(
                    f"open_y_{r}_{c}", b[1][1],
                    minimum=FRONT_Y1 + TRAVEL - 0.005,
                    maximum=FRONT_Y1 + TRAVEL + 0.005,
                    unit="m",
                )

    ctx.expect_metric("front_height", FRONT_H, minimum=0.20, maximum=0.26, unit="m")

    ctx.attach_artifact("qa/previews/front.png", name="front", caption="Front view.")
    ctx.attach_artifact("qa/previews/three_quarter.png", name="three_quarter",
                        caption="Three-quarter view.")
    ctx.attach_artifact("qa/previews/drawer_open.png", name="drawer_open",
                        caption="A drawer pulled open.")
    return ctx.report()
