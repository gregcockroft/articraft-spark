"""White painted chest of drawers: 3 x 3 fluted drawers on prismatic slides.

Conventions
-----------
* metres, Z up, drawer fronts face +Y (carcass front plane at y = +0.225)
* carcass = 2 side panels + top + bottom + back + 2 vertical dividers + 2
  horizontal boards, which carve the interior into nine separate openings
* every drawer is its own rigid body: fluted front, box, brass bar pull and four
  brass corner brackets. It rests on the board beneath its opening and slides
  straight out on +Y through its own prismatic joint.
"""

from __future__ import annotations

from build123d import Box, Cylinder, Polygon, Pos, Rot, Vector, extrude
from articraft.sdk import (
    ExtrudeGeometry,
    JointAxis,
    JointDOF,
    Material,
    RigidBodyAssembly,
    TestContext,
    TestReport,
)

# ---------------------------------------------------------------- dimensions

W_OUT = 1.20             # overall width            (X: -0.60 .. +0.60)
D_OUT = 0.45             # overall depth            (Y: -0.225 .. +0.225)
H_OUT = 0.90             # overall height, feet in  (Z: 0 .. 0.90)

FOOT_H = 0.055           # four short bracket feet
CAR_MIN = FOOT_H         # lowest carcass board
T_TOP = 0.038
CAR_MAX = H_OUT - T_TOP  # underside of the top board

T_SIDE = 0.020           # side panels
T_BOARD = 0.018          # bottom board / horizontal boards
T_DIV = 0.016            # vertical dividers
T_BACK = 0.012           # back panel

FRONT = D_OUT / 2.0      # +0.225 carcass front plane
BACK = -D_OUT / 2.0

INTERIOR_X0 = -W_OUT / 2 + T_SIDE
INTERIOR_X1 = W_OUT / 2 - T_SIDE
INTERIOR_Z0 = CAR_MIN + T_BOARD
INTERIOR_Z1 = CAR_MAX
INTERIOR_Y0 = BACK + T_BACK
INTERIOR_Y1 = FRONT

INTERIOR_W = INTERIOR_X1 - INTERIOR_X0
INTERIOR_H = INTERIOR_Z1 - INTERIOR_Z0
INTERIOR_D = INTERIOR_Y1 - INTERIOR_Y0

COL_W = (INTERIOR_W - 2 * T_DIV) / 3.0      # drawer opening width
ROW_H = (INTERIOR_H - 2 * T_BOARD) / 3.0    # drawer opening height

# fluted front panel --------------------------------------------------------
GAP = 0.004              # reveal between a front and its opening edge
FRONT_BACK = 0.2090      # back face of a front panel
FRONT_ROOT = 0.2220      # floor of a flute groove
FRONT_CREST = 0.2275     # crest of a flute (2.5 mm proud of the carcass)

PANEL_W = COL_W - 2 * GAP
PANEL_H = ROW_H - 2 * GAP

FLUTE_PERIOD = 0.008
FLUTE_MARGIN = 0.008     # flat band along each vertical edge of the front
FLUTE_LAND = 0.0040      # flat crest width
FLUTE_FLOOR = 0.0018     # flat groove-floor width

# drawer box ----------------------------------------------------------------
BOX_SIDE_T = 0.012       # box sides double as the runners
BOX_BACK_T = 0.012
BOX_CLEAR = 0.006        # clearance between box side and divider
BOX_H = 0.190
BOX_FRONT_Y = FRONT_BACK + 0.008             # box rails bite 8 mm into the front
BOX_BACK_Y = -0.150

TRAVEL = 0.28            # drawer pull-out

WHITE = (0.930, 0.918, 0.876)
WOOD_RAW = (0.78, 0.64, 0.47)
BRASS = Material.STEEL.but(
    name="brass", density=8500.0, color=(0.62, 0.47, 0.20), metallic=1.0, roughness=0.30
)
PAINT = Material.HARDWOOD.but(name="painted_wood", roughness=0.45)
WOOD = Material.HARDWOOD


def row_z(row: int) -> tuple[float, float]:
    """Floor and ceiling Z of one drawer opening (row 0 is the bottom row)."""
    floor = INTERIOR_Z0 + row * (ROW_H + T_BOARD)
    return floor, floor + ROW_H


def col_x(col: int) -> float:
    """Centre X of one drawer opening (column 0 is the left column)."""
    return INTERIOR_X0 + COL_W / 2 + col * (COL_W + T_DIV)


# ------------------------------------------------------------------- pieces


def fluted_front(width: float, height: float) -> ExtrudeGeometry:
    """Corrugated panel: flat back, fine vertical flutes across the face.

    The profile lies in the XY plane (X across the drawer, Y toward the viewer)
    and is extruded along Z, so each flute runs vertically. One profile gives a
    flat back face and a constant-thickness fluted face.
    """
    back_y = FRONT_BACK - FRONT_CREST          # relative to the crest plane
    root_y = FRONT_ROOT - FRONT_CREST

    x_left = -width / 2 + FLUTE_MARGIN
    x_right = width / 2 - FLUTE_MARGIN
    periods = max(6, int(round((x_right - x_left) / FLUTE_PERIOD)))
    step = (x_right - x_left) / periods
    slope = (step - FLUTE_LAND - FLUTE_FLOOR) / 2.0

    # crest boundary run left to right: land, slope down, groove floor, slope up
    rib: list[tuple[float, float]] = [(-width / 2, 0.0), (x_left, 0.0)]
    for index in range(periods):
        x = x_left + index * step
        rib.append((x, 0.0))
        rib.append((x + FLUTE_LAND, 0.0))
        rib.append((x + FLUTE_LAND + slope, root_y))
        rib.append((x + FLUTE_LAND + slope + FLUTE_FLOOR, root_y))
    rib.append((x_right, 0.0))
    rib.append((width / 2, 0.0))

    return ExtrudeGeometry([(width / 2, back_y), (-width / 2, back_y)] + rib, height, center=True)


def corner_bracket(arm: float, band: float, thickness: float, corner_x: float, corner_z: float,
                   y_back: float, sx: int, sz: int):
    """L-shaped brass plate flat on the front, legs running along both edges.

    Authored in the front plane (u across, v up) and laid down on +Y, so the
    plate stands a little proud of the flutes and hugs the two neighbouring edges.
    """
    pts = [
        (0.0, 0.0),
        (-sx * arm, 0.0),
        (-sx * arm, sz * band),
        (-sx * band, sz * band),
        (-sx * band, sz * arm),
        (0.0, sz * arm),
    ]
    plate = extrude(Polygon(*pts), amount=thickness, dir=Vector(0, 0, 1))
    return Pos(corner_x, y_back, corner_z) * Rot(-90, 0, 0) * plate


def bracket_foot(outer_y: float, thickness: float, x_inner: float):
    """Stepped bracket foot: silhouette in the vertical side plane, extruded in X."""
    inward = -1.0 if outer_y > 0 else 1.0
    far = outer_y + inward * 0.075             # inner toe of the foot
    pts = [
        (0.000, outer_y),
        (0.000, far),
        (0.026, far),
        (0.026, outer_y + inward * 0.058),
        (0.044, outer_y + inward * 0.058),
        (0.062, outer_y + inward * 0.034),
        (0.062, outer_y),
    ]
    solid = extrude(Polygon(*pts), amount=thickness, dir=Vector(0, 0, -1))
    return Pos(x_inner, 0.0, 0.0) * Rot(0, -90, 0) * solid


def add_drawer(model: RigidBodyAssembly, row: int, col: int) -> None:
    body = model.rigid_body(f"drawer_{row}_{col}")
    cx = col_x(col)
    z_floor, z_ceiling = row_z(row)
    panel_z = (z_floor + z_ceiling) / 2.0

    # -- fluted front panel
    front = fluted_front(PANEL_W, PANEL_H)
    front.translate(cx, FRONT_CREST, panel_z)
    body.add(front, name="front", material=PAINT, color=WHITE)

    # -- drawer box: bottom, two runner sides, back; all bite into the front
    box_w = COL_W - 2 * BOX_CLEAR
    box_depth = BOX_FRONT_Y - BOX_BACK_Y
    cy = (BOX_FRONT_Y + BOX_BACK_Y) / 2.0

    body.add(
        Pos(cx, cy, z_floor + BOX_SIDE_T / 2) * Box(box_w, box_depth, BOX_SIDE_T),
        name="box_bottom", material=WOOD, color=WOOD_RAW,
    )
    for tag, sign in (("left", -1), ("right", 1)):
        body.add(
            Pos(cx + sign * (box_w - BOX_SIDE_T) / 2, cy, z_floor + BOX_H / 2)
            * Box(BOX_SIDE_T, box_depth, BOX_H),
            name=f"box_{tag}", material=WOOD, color=WOOD_RAW,
        )
    body.add(
        Pos(cx, BOX_BACK_Y + BOX_BACK_T / 2, z_floor + BOX_H / 2)
        * Box(box_w, BOX_BACK_T, BOX_H),
        name="box_back", material=WOOD, color=WOOD_RAW,
    )

    # -- small brass bar pull standing proud of the front centre
    bar_y = FRONT_CREST + 0.024
    bar_len = 0.075
    body.add(
        Pos(cx, bar_y, panel_z) * Cylinder(0.006, bar_len),
        name="pull_bar", material=BRASS,
    )
    post_root = FRONT_CREST - 0.004            # sunk into the flute crests
    post_h = bar_y - post_root
    for tag, dz in (("lower", -0.027), ("upper", 0.027)):
        body.add(
            Pos(cx, post_root + post_h / 2, panel_z + dz) * Rot(-90, 0, 0)
            * Cylinder(0.0045, post_h),
            name=f"pull_{tag}_post", material=BRASS,
        )

    # -- four brass corner brackets on the flat margins of the front
    arm, band = 0.058, 0.013
    y_back = FRONT_CREST - 0.004               # embedded in the crests
    for tag, sx, sz in (("bl", -1, -1), ("br", 1, -1), ("tl", -1, 1), ("tr", 1, 1)):
        body.add(
            corner_bracket(
                arm, band, 0.0075,
                cx + sx * PANEL_W / 2, panel_z + sz * PANEL_H / 2, y_back, sx, sz,
            ),
            name=f"bracket_{tag}", material=BRASS,
        )


def build_object_model() -> RigidBodyAssembly:
    model = RigidBodyAssembly("chest_of_drawers")
    carcass = model.rigid_body("carcass")

    mid_y = (BACK + FRONT) / 2.0
    depth = FRONT - BACK
    car_h = CAR_MAX - CAR_MIN
    car_z = (CAR_MIN + CAR_MAX) / 2.0

    # two side panels
    for tag, sx in (("left", -1), ("right", 1)):
        carcass.add(
            Pos(sx * (W_OUT / 2 - T_SIDE / 2), mid_y, car_z) * Box(T_SIDE, depth, car_h),
            name=f"side_{tag}", material=PAINT, color=WHITE,
        )

    # top board: flush with the back, cornice overhang at the front and sides
    top_w, top_d = W_OUT + 0.030, D_OUT + 0.015
    carcass.add(
        Pos(0.0, BACK + top_d / 2, CAR_MAX + T_TOP / 2) * Box(top_w, top_d, T_TOP),
        name="top", material=PAINT, color=WHITE,
    )

    # bottom board
    carcass.add(
        Pos(0.0, mid_y, CAR_MIN + T_BOARD / 2) * Box(W_OUT - 2 * T_SIDE + 0.004, depth, T_BOARD),
        name="bottom", material=PAINT, color=WHITE,
    )

    # back panel, tucked into sides and boards
    carcass.add(
        Pos(0.0, BACK + T_BACK / 2, car_z) * Box(INTERIOR_W + 0.008, T_BACK, car_h + 0.012),
        name="back", material=PAINT, color=WHITE,
    )

    # two vertical dividers: three equal chutes
    for tag, x in (
        ("left", INTERIOR_X0 + COL_W + T_DIV / 2),
        ("right", INTERIOR_X1 - COL_W - T_DIV / 2),
    ):
        carcass.add(
            Pos(x, mid_y + T_BACK / 2, car_z) * Box(T_DIV, INTERIOR_D - 0.002, INTERIOR_H),
            name=f"divider_{tag}", material=PAINT, color=WHITE,
        )

    # horizontal boards: three stacked rows of openings
    for row in (1, 2):
        z_lo = INTERIOR_Z0 + row * ROW_H + (row - 1) * T_BOARD
        carcass.add(
            Pos(0.0, mid_y + T_BACK / 2, z_lo + T_BOARD / 2)
            * Box(INTERIOR_W + 0.004, INTERIOR_D - 0.002, T_BOARD),
            name=f"rail_row{row}", material=PAINT, color=WHITE,
        )

    # four short stepped bracket feet under the corners
    foot_t = 0.032
    foot_x = W_OUT / 2 - foot_t                # foot reaches the outer face
    right_front = bracket_foot(FRONT - 0.002, foot_t, foot_x)
    right_back = bracket_foot(BACK + 0.002, foot_t, foot_x)
    carcass.add(right_front, name="foot_right_front", material=PAINT, color=WHITE)
    carcass.add(right_back, name="foot_right_back", material=PAINT, color=WHITE)
    carcass.add(Rot(0, 0, 180) * right_back, name="foot_left_front", material=PAINT, color=WHITE)
    carcass.add(Rot(0, 0, 180) * right_front, name="foot_left_back", material=PAINT, color=WHITE)

    # drawers and one horizontal prismatic slide for each
    for row in range(3):
        for col in range(3):
            add_drawer(model, row, col)
            drawer = model.get_rigid_body(f"drawer_{row}_{col}")
            z_floor, z_ceiling = row_z(row)
            anchor = (col_x(col), FRONT, (z_floor + z_ceiling) / 2.0)
            model.joint(
                f"slide_{row}_{col}",
                carcass.at(anchor),
                drawer.at(anchor),
                dofs=(JointDOF(JointAxis.TRANS_Y, limits=(0.0, TRAVEL)),),
            )

    model.articulation(
        "drawers",
        root=carcass,
        joints=[f"slide_{r}_{c}" for r in range(3) for c in range(3)],
    )
    return model


object_model = build_object_model()


def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    overall = ctx.measure_geometry()
    case = ctx.measure_geometry("carcass")

    ctx.expect_metric("width", overall.dimensions[0], minimum=1.19, maximum=1.30, unit="m",
                      details="overall width incl. the top overhang")
    ctx.expect_metric("case_depth", case.dimensions[1], minimum=0.44, maximum=0.47, unit="m",
                      details="carcass depth with the fronts closed")
    ctx.expect_metric("height", overall.dimensions[2], minimum=0.89, maximum=0.91, unit="m",
                      details="height incl. feet and top board")

    ctx.check("nine_drawer_bodies", len(object_model.rigid_bodies) == 10,
              f"{len(object_model.rigid_bodies)} bodies: carcass + 9 drawers")
    ctx.check("nine_prismatic_joints", len(object_model.joints) == 9,
              f"{len(object_model.joints)} joints")
    ctx.check(
        "slides_run_along_y",
        all([d.axis for d in j.dofs] == [JointAxis.TRANS_Y] for j in object_model.joints),
        "every drawer joint is one translation along the drawer axis",
    )

    # fronts stand proud of the carcass front plane, inside the case footprint
    crest_y = ctx.shape_world_bounds("drawer_1_1", "front")[1][1]
    ctx.expect_metric("front_proud", crest_y - FRONT, minimum=0.001, maximum=0.006, unit="m",
                      details="flute crests proud of the carcass front plane")
    ctx.expect_within("drawer_1_1", "carcass", inner_shape="box_bottom", outer_shape="top",
                      axes="x", name="box_stays_inside_the_case")
    ctx.expect_distance("drawer_1_1", "drawer_1_0", min_distance=0.002,
                        name="neighbouring_fronts_keep_their_reveal")

    # each drawer rests on the board beneath it - real support, not floating
    ctx.expect_contact("drawer_0_0", "carcass", shape_a="box_bottom", shape_b="bottom",
                       contact_tol=0.0005)
    ctx.expect_contact("drawer_1_1", "carcass", shape_a="box_bottom", shape_b="rail_row1",
                       contact_tol=0.0005)
    ctx.expect_contact("drawer_2_2", "carcass", shape_a="box_bottom", shape_b="rail_row2",
                       contact_tol=0.0005)

    # the box stays clear of the dividers that bound its own opening
    ctx.expect_distance("drawer_1_1", "carcass", shape_a="box_left", shape_b="divider_left",
                        min_distance=0.002)
    ctx.expect_distance("drawer_1_1", "carcass", shape_a="box_right", shape_b="divider_right",
                        min_distance=0.002)

    # brass hardware stands proud of the fluted face
    bar_root_y = ctx.shape_world_bounds("drawer_1_1", "pull_bar")[0][1]
    ctx.check("pull_stands_proud", bar_root_y - crest_y > 0.012,
              f"bar root is {bar_root_y - crest_y:.4f} m in front of the flutes")
    bracket_y = ctx.shape_world_bounds("drawer_1_1", "bracket_tl")[1][1]
    ctx.check("brackets_stand_proud", bracket_y - crest_y > 0.0015,
              f"brackets stand {bracket_y - crest_y:.4f} m proud of the flutes")

    # fine vertical flutes across every front
    ctx.record_metric("flute_period", FLUTE_PERIOD, unit="m")
    ctx.record_metric("flute_depth", FRONT_CREST - FRONT_ROOT, unit="m")
    ctx.record_metric("flutes_per_front", round(PANEL_W / FLUTE_PERIOD), unit="count")

    # motion: a drawer swept through its slide stays clear of the case
    poses = ctx.sample_joint("slide_1_1", samples=5)
    ctx.expect_no_collision_at_poses("drawer_1_1", "carcass", poses, shape_a="front",
                                     name="front_clear_through_its_reveal")
    ctx.expect_no_collision_at_poses("drawer_1_1", "carcass", poses, shape_a="box_left",
                                     name="box_left_clear_of_the_case")
    ctx.expect_no_collision_at_poses("drawer_1_1", "carcass", poses, shape_a="box_right",
                                     name="box_right_clear_of_the_case")
    ctx.expect_no_collision_at_poses("drawer_1_1", "carcass", poses, shape_a="box_back",
                                     name="box_back_clear_of_the_case")
    # ...and it keeps riding the board under its opening the whole way
    ctx.expect_contact_at_poses("drawer_1_1", "carcass", poses, shape_a="box_bottom",
                                shape_b="rail_row1", contact_tol=0.0005,
                                name="drawer_keeps_riding_its_rail")
    handle = ctx.track_point("drawer_1_1", (0.0, FRONT_CREST, 0.0), poses)
    ctx.check(
        "drawer_moves_straight_out",
        handle[-1][1] - handle[0][1] > TRAVEL - 0.002
        and abs(handle[-1][0] - handle[0][0]) < 1e-6
        and abs(handle[-1][2] - handle[0][2]) < 1e-6,
        f"handle travels {handle[-1][1] - handle[0][1]:.4f} m straight along +Y",
    )

    # a pulled drawer keeps riding the board under its opening
    with ctx.pose({"slide_1_1.transY": TRAVEL}):
        pulled = ctx.shape_world_bounds("drawer_1_1", "box_bottom")
        ctx.check(
            "pulled_box_still_on_rail",
            abs(pulled[0][2] - row_z(1)[0]) < 1e-6,
            f"box bottom stays at rail level z={pulled[0][2]:.4f} when pulled out",
        )

    ctx.record_metric("drawer_opening_w", COL_W, unit="m")
    ctx.record_metric("drawer_opening_h", ROW_H, unit="m")
    ctx.record_metric("drawer_travel", TRAVEL, unit="m")
    ctx.attach_artifact("qa/previews/01_three_quarter.png", name="three quarter view",
                        caption="White dresser, 3x3 fluted fronts, brass pulls and corner brackets.")
    ctx.attach_artifact("qa/previews/04_section_x.png", name="carcass section",
                        caption="Sides, top, bottom, back, dividers and rails around nine openings.")
    ctx.attach_artifact("qa/previews/06_drawers_open.png", name="drawers pulled out",
                        caption="Three drawers slid out on their own prismatic joints.")
    ctx.attach_artifact("qa/previews/05_front_closeup.png", name="front close-up",
                        caption="Vertical flutes, brass bar pull and corner brackets.")
    return ctx.report()
