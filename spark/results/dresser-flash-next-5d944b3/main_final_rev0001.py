"""A white painted chest of drawers: nine fluted drawers in a 3x3 grid.

Coordinates: Z is up, the drawer fronts face +Y, the carcass is centred on X=0.
All lengths are meters, all build123d rotations are degrees.

Construction
    carcass  : two side panels, bottom board, back panel, two vertical dividers,
               horizontal rails, overhanging top, four bracket feet, aprons and
               drawer runners, so each drawer sits in its own opening.
    drawer   : a real box (sides, back, bottom) behind an overlay front panel.
               The front carries a reeded (fluted) field cut as one extruded
               profile, a brass bar handle on stand-off posts, and a brass
               corner bracket at each of its four corners.
    motion   : one prismatic slide per drawer (nine joints) straight out along
               +Y; the drawer bottoms rest on the wooden runners.
"""

from __future__ import annotations

from build123d import Box, Cylinder, Polygon, Pos, Rot, extrude

from articraft.sdk import (
    JointAxis,
    JointDOF,
    Material,
    RigidBodyAssembly,
    TestContext,
    TestReport,
)

# --------------------------------------------------------------- measurements
WIDTH = 1.200  # overall, across the top board
DEPTH = 0.464  # overall, back of the top board to its front edge
HEIGHT = 0.900  # overall, top of the feet to the top board

FEET_H = 0.080
SIDE_T = 0.017
BACK_T = 0.012
DIV_T = 0.016
RAIL_T = 0.018
BOTTOM_T = 0.020
TOP_T = 0.026

CARCASS_X = 0.563  # outer half width of the carcass; the top overhangs to 0.600
Y_BACK = 0.0
Y_FRONT = 0.430  # front frame plane of the carcass

BOTTOM_TOP = FEET_H + BOTTOM_T  # 0.100, top of the bottom board
ROW1 = BOTTOM_TOP + RAIL_T  # 0.118, bottom of the lowest openings
TOP_RAIL_BOTTOM = 0.856
TOP_Z = TOP_RAIL_BOTTOM + RAIL_T  # 0.874, underside of the top board
ROW_PITCH = 0.252
OPEN_H = ROW_PITCH - RAIL_T  # 0.234
OPEN_W = (2 * (CARCASS_X - SIDE_T) - 2 * DIV_T) / 3  # 0.353333 clear opening
COL_PITCH = OPEN_W + DIV_T  # 0.369333 between column centre lines

FRONT_T = 0.014
FRONT_BACK = Y_FRONT + 0.0015  # overlay fronts clear the frame by 1.5 mm
FRONT_FRONT = FRONT_BACK + FRONT_T
FRONT_W = COL_PITCH - 0.006
FRONT_H = ROW_PITCH - 0.006

FLUTE_PITCH = 0.0095
FLUTE_MARGIN_X = 0.027  # flat border left around the reeded field
FLUTE_MARGIN_Z = 0.025
FLUTE_ROOT = FRONT_BACK + 0.003  # embedded in the front panel
FLUTE_GROOVE = FRONT_FRONT + 0.0005  # groove floor sits just proud of the face
FLUTE_CREST = FRONT_FRONT + 0.0040  # reeded field stands 4 mm proud

RUNNER_T = 0.010
RUNNER_BITE = 0.003  # runner is glued to the divider / side over its length
RUNNER_GAP = OPEN_W - 2 * RUNNER_T  # 0.333333 between the two runner faces
BOX_SIDE_T = 0.012
BOX_BACK_T = 0.013
BOX_BOTTOM_T = 0.009
BOX_SIDE_H = 0.170
BOX_W = RUNNER_GAP - 0.002  # drawer sides: 1 mm sliding clearance per side
BOX_BOTTOM_W = RUNNER_GAP + 0.010  # bottom overlaps each runner by 5 mm
BOX_BACK_Y0 = 0.075
BOX_FRONT_Y = FRONT_BACK + 0.006  # box parts bite 6 mm into the front panel
TRAVEL = 0.255

HANDLE_BAR_Y = FRONT_FRONT + 0.0260
HANDLE_BAR_LEN = 0.100
HANDLE_BAR_R = 0.0055
HANDLE_POST_R = 0.0035
HANDLE_RISE = 0.040

BRACKET_LEN = 0.060
BRACKET_W = 0.013
BRACKET_T = 0.0030
BRACKET_INSET = 0.005
BRACKET_Y = FRONT_FRONT + 0.0005  # plate centre: 1 mm buried, 2 mm proud
BRACKET_TIP = 0.008

PAINT_WHITE = (0.920, 0.908, 0.870)
PAINT_INNER = (0.865, 0.852, 0.815)
PAINT_REED = (0.940, 0.930, 0.900)
BRASS_COLOR = (0.760, 0.580, 0.245)

PAINTED_WOOD = Material.HARDWOOD.but(name="painted_wood", roughness=0.55)
BRASS = Material(name="brass", density=8500.0, friction=(0.35, 0.30))


def slab(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float):
    """Axis-aligned box described by its minimum and maximum corner."""
    return Pos((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2) * Box(
        x1 - x0, y1 - y0, z1 - z0
    )


def column_x(col: int) -> float:
    return (col - 2) * COL_PITCH


def opening_bottom(row: int) -> float:
    return ROW1 + (row - 1) * ROW_PITCH


def row_z(row: int) -> float:
    return opening_bottom(row) + OPEN_H / 2


def bracket_profile(sign_y: float):
    """Bracket foot profile in the YZ plane; local Y points into the carcass."""
    depth = 0.100
    top = FEET_H + 0.012
    sweep = 0.046  # where the concave scoop meets the floor
    arc = 0.030
    pts = [
        (depth, 0.0),
        (depth, top),
        (0.0, top),
        (0.0, top - 0.020),
        (arc, top - 0.020),
    ]
    # Chamfered bottom edge running back to the toe of the foot.
    pts += [(arc + 0.010, 0.020), (arc + sweep * 0.55, 0.004), (arc + sweep, 0.0)]
    if sign_y > 0:  # front foot: full-height post at the front face
        pts = [(y, z) for y, z in pts]
    return pts


def foot(sign_x: float, sign_y: float):
    """Bracket foot: full-height outer post with a swept-in inner cheek."""
    x0, x1 = (
        (CARCASS_X - 0.060, CARCASS_X)
        if sign_x > 0
        else (-CARCASS_X, -CARCASS_X + 0.060)
    )
    top = FEET_H + 0.012
    profile = bracket_profile(sign_y)
    if sign_y < 0:  # back foot: post at the back face, cheek swept toward +Y
        profile = [(Y_BACK + y, z) for y, z in profile]
        y0, y1 = Y_BACK, Y_BACK + 0.100
    else:  # front foot: post at the front face, cheek swept toward -Y
        profile = [(Y_FRONT - y, z) for y, z in profile]
        y0, y1 = Y_FRONT - 0.100, Y_FRONT
    pts = [(x0, u, v) for u, v in profile]
    return extrude(Polygon(*pts), amount=x1 - x0, dir=(1, 0, 0))


def fluted_field(width: float, height: float):
    """Reeded panel: profile in the panel's XY cross section, extruded up in Z."""
    n = max(1, round(width / FLUTE_PITCH))
    p = width / n
    groove = FLUTE_GROOVE - FLUTE_ROOT
    crest = FLUTE_CREST - FLUTE_ROOT
    pts: list[tuple[float, float]] = [(0.0, 0.0)]
    for i in range(n):
        x = i * p
        pts += [(x, groove), (x + 0.34 * p, groove), (x + 0.50 * p, crest),
                (x + 0.84 * p, crest)]
    pts += [(width, groove), (width, 0.0)]
    return extrude(Polygon(*pts), amount=height, dir=(0, 0, 1))


def corner_bracket(sign_x: float, sign_z: float):
    """Brass corner plate with pointed leg ends, in the front-face plane.

    The plate is authored at the drawer-front centre origin: local (u, v) are the
    front's X and Z with the corner at the origin and the legs running into the
    front. The plate is extruded along +Y.
    """
    l, w, tip = BRACKET_LEN, BRACKET_W, BRACKET_TIP
    pts = [
        (0.0, 0.0),
        (0.0, -(l - tip)),
        (-w / 2, -l),
        (-w, -(l - tip)),
        (-w, -w),
        (-(l - tip), -w),
        (-l, -w / 2),
        (-(l - tip), 0.0),
    ]
    profile = [(sign_x * u, 0.0, sign_z * v) for u, v in pts]
    return extrude(Polygon(*profile), amount=BRACKET_T, dir=(0, 1, 0))


def build_object_model() -> RigidBodyAssembly:
    model = RigidBodyAssembly("fluted_chest_of_drawers")

    # ----------------------------------------------------------------- carcass
    carcass = model.rigid_body("carcass")

    def paint(shape, name: str, color=PAINT_WHITE):
        return carcass.add(shape, name=name, material=PAINTED_WOOD, color=color)

    for sign in (-1, 1):
        x0, x1 = (
            (CARCASS_X - SIDE_T, CARCASS_X)
            if sign > 0
            else (-CARCASS_X, -CARCASS_X + SIDE_T)
        )
        paint(
            slab(x0, x1, Y_BACK, Y_FRONT, FEET_H, TOP_Z + 0.003),
            f"side_panel_{'right' if sign > 0 else 'left'}",
        )

    paint(
        slab(-CARCASS_X - 0.002, CARCASS_X + 0.002, Y_BACK, Y_BACK + BACK_T,
             FEET_H, TOP_Z + 0.003),
        "back_panel",
        PAINT_INNER,
    )
    paint(
        slab(-CARCASS_X - 0.002, CARCASS_X + 0.002, Y_BACK, Y_FRONT, FEET_H, BOTTOM_TOP),
        "bottom_board",
        PAINT_INNER,
    )

    # Horizontal rails: bottom, two middle, and the top rail under the top board.
    for index, row in enumerate((1, 2), start=1):
        z0 = opening_bottom(row) + OPEN_H
        paint(
            slab(-CARCASS_X, CARCASS_X, Y_BACK + BACK_T, Y_FRONT, z0, z0 + RAIL_T),
            f"mid_rail_{index}",
            PAINT_INNER,
        )
    paint(
        slab(-CARCASS_X, CARCASS_X, Y_BACK + BACK_T, Y_FRONT, BOTTOM_TOP, ROW1),
        "bottom_rail",
        PAINT_INNER,
    )
    paint(
        slab(-CARCASS_X, CARCASS_X, Y_BACK + BACK_T, Y_FRONT, TOP_RAIL_BOTTOM, TOP_Z),
        "top_rail",
        PAINT_INNER,
    )

    # Two full-height dividers make the three columns.
    for sign in (-1, 1):
        centre = sign * (OPEN_W / 2 + DIV_T / 2)
        paint(
            slab(centre - DIV_T / 2, centre + DIV_T / 2, Y_BACK + BACK_T, Y_FRONT,
                 BOTTOM_TOP - 0.003, TOP_RAIL_BOTTOM + 0.003),
            f"divider_{'left' if sign < 0 else 'right'}",
        )

    # Overhanging top board with a shadow moulding under its front edge.
    paint(
        slab(-WIDTH / 2, WIDTH / 2, -0.012, 0.452, TOP_Z, TOP_Z + TOP_T),
        "top",
    )
    paint(
        slab(-0.592, 0.592, 0.398, 0.437, TOP_RAIL_BOTTOM + 0.008, TOP_Z),
        "top_moulding",
    )

    # Bracket feet and the recessed aprons that run between them.
    for sx in (-1, 1):
        for sy in (-1, 1):
            paint(
                foot(sx, sy),
                f"foot_{'front' if sy > 0 else 'back'}_{'right' if sx > 0 else 'left'}",
            )
    paint(
        slab(-0.505, 0.505, Y_BACK + 0.018, Y_BACK + 0.036, 0.028, FEET_H - 0.010),
        "apron_back",
        PAINT_INNER,
    )
    paint(
        slab(-0.505, 0.505, Y_FRONT - 0.036, Y_FRONT - 0.018, 0.028, FEET_H - 0.010),
        "apron_front",
        PAINT_INNER,
    )
    for sx in (-1, 1):
        x0, x1 = (
            (CARCASS_X - 0.036, CARCASS_X - 0.018)
            if sx > 0
            else (-CARCASS_X + 0.018, -CARCASS_X + 0.036)
        )
        paint(
            slab(x0, x1, 0.090, Y_FRONT - 0.090, 0.028, FEET_H - 0.010),
            f"apron_side_{'right' if sx > 0 else 'left'}",
            PAINT_INNER,
        )

    # Drawer runners: each opening has a strip on both cheeks.
    for row in (1, 2, 3):
        z0 = opening_bottom(row)
        for col in (1, 2, 3):
            cx = column_x(col)
            inner = cx - OPEN_W / 2
            outer = cx + OPEN_W / 2
            paint(
                slab(inner - RUNNER_BITE, inner + RUNNER_T, Y_BACK + BACK_T, Y_FRONT,
                     z0, z0 + RUNNER_T),
                f"runner_r{row}c{col}_left",
                PAINT_INNER,
            )
            paint(
                slab(outer - RUNNER_T, outer + RUNNER_BITE, Y_BACK + BACK_T, Y_FRONT,
                     z0, z0 + RUNNER_T),
                f"runner_r{row}c{col}_right",
                PAINT_INNER,
            )

    # ---------------------------------------------------------------- drawers
    field_w = FRONT_W - 2 * FLUTE_MARGIN_X
    field_h = FRONT_H - 2 * FLUTE_MARGIN_Z
    field = fluted_field(field_w, field_h)
    joints: list[str] = []

    for row in (1, 2, 3):
        for col in (1, 2, 3):
            cx = column_x(col)
            cz = row_z(row)
            box_bottom = opening_bottom(row) + RUNNER_T

            drawer = model.rigid_body(f"drawer_r{row}c{col}")

            drawer.add(
                slab(cx - FRONT_W / 2, cx + FRONT_W / 2, FRONT_BACK, FRONT_FRONT,
                     cz - FRONT_H / 2, cz + FRONT_H / 2),
                name="front_panel",
                material=PAINTED_WOOD,
                color=PAINT_WHITE,
            )
            drawer.add(
                Pos(cx - field_w / 2, FLUTE_ROOT, cz - field_h / 2) * field,
                name="flutes",
                material=PAINTED_WOOD,
                color=PAINT_REED,
            )

            half = BOX_W / 2
            bottom_half = BOX_BOTTOM_W / 2
            for sign in (-1, 1):
                x0, x1 = (
                    (cx + half - BOX_SIDE_T, cx + half)
                    if sign > 0
                    else (cx - half, cx - half + BOX_SIDE_T)
                )
                drawer.add(
                    slab(x0, x1, BOX_BACK_Y0, BOX_FRONT_Y, box_bottom,
                         box_bottom + BOX_SIDE_H),
                    name=f"side_{'right' if sign > 0 else 'left'}",
                    material=PAINTED_WOOD,
                    color=PAINT_INNER,
                )
            drawer.add(
                slab(cx - half, cx + half, BOX_BACK_Y0, BOX_BACK_Y0 + BOX_BACK_T,
                     box_bottom, box_bottom + BOX_SIDE_H),
                name="back_panel",
                material=PAINTED_WOOD,
                color=PAINT_INNER,
            )
            drawer.add(
                slab(cx - bottom_half, cx + bottom_half,
                     BOX_BACK_Y0 + 0.004, BOX_FRONT_Y,
                     box_bottom, box_bottom + BOX_BOTTOM_T),
                name="bottom",
                material=PAINTED_WOOD,
                color=PAINT_INNER,
            )

            # Brass bar handle on two stand-off posts.
            drawer.add(
                Pos(cx, HANDLE_BAR_Y, cz) * Cylinder(HANDLE_BAR_R, HANDLE_BAR_LEN),
                name="handle_bar",
                material=BRASS,
                color=BRASS_COLOR,
            )
            post_y0 = FRONT_BACK + 0.003
            for sign in (-1, 1):
                drawer.add(
                    Pos(cx, (post_y0 + HANDLE_BAR_Y) / 2, cz + sign * HANDLE_RISE)
                    * Rot(90, 0, 0)
                    * Cylinder(HANDLE_POST_R, HANDLE_BAR_Y - post_y0),
                    name=f"handle_post_{'top' if sign > 0 else 'bottom'}",
                    material=BRASS,
                    color=BRASS_COLOR,
                )

            # Brass corner brackets on the flat border of the front.
            for sx in (-1, 1):
                for sz in (-1, 1):
                    bracket = corner_bracket(sx, sz)
                    drawer.add(
                        Pos(
                            cx + sx * (FRONT_W / 2 - BRACKET_INSET),
                            BRACKET_Y - BRACKET_T / 2,
                            cz + sz * (FRONT_H / 2 - BRACKET_INSET),
                        )
                        * bracket,
                        name=f"corner_bracket_x{sx}_z{sz}",
                        material=BRASS,
                        color=BRASS_COLOR,
                    )

            joint_name = f"drawer_slide_r{row}c{col}"
            model.joint(
                joint_name,
                carcass.at((cx, Y_FRONT, cz)),
                drawer.at((cx, Y_FRONT, cz)),
                dofs=(JointDOF(JointAxis.TRANS_Y, limits=(0.0, TRAVEL)),),
            )
            joints.append(joint_name)

    model.articulation("drawers", root=carcass, joints=joints)
    return model


object_model = build_object_model()


def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    drawers = [f"drawer_r{row}c{col}" for row in (1, 2, 3) for col in (1, 2, 3)]

    # ------------------------------------------------------------ overall form
    lo, hi = ctx.part_world_bounds("carcass")
    ctx.record_metric("width", round(hi[0] - lo[0], 4), unit="m")
    ctx.record_metric("depth", round(hi[1] - lo[1], 4), unit="m")
    ctx.record_metric("height", round(hi[2] - lo[2], 4), unit="m")
    ctx.record_metric("drawer_count", len(drawers))
    ctx.check(
        "envelope_1p20_wide_0p46_deep_0p90_tall",
        abs(hi[0] - lo[0] - WIDTH) < 0.004
        and abs(hi[1] - lo[1] - DEPTH) < 0.008
        and abs(hi[2] - lo[2] - HEIGHT) < 0.002,
        f"carcass bounds {tuple(round(v, 3) for v in lo)} .. "
        f"{tuple(round(v, 3) for v in hi)}",
    )

    # ------------------------------------------------------- joints and motion
    slides = list(object_model.joints)
    ctx.check(
        "nine_prismatic_joints",
        len(slides) == 9 and all(
            len(j.dofs) == 1 and j.dofs[0].axis is JointAxis.TRANS_Y for j in slides
        ),
        f"{len(slides)} joints, each one free slide along +Y",
    )
    travel = tuple(sorted({round(j.dofs[0].limits[1], 3) for j in slides}))
    ctx.record_metric("drawer_travel", travel[-1], unit="m")
    tree = next(a for a in object_model.articulations if a.name == "drawers")
    ctx.check(
        "articulation_tree_has_nine_slides",
        len(tree.joints) == 9 and tree.root is object_model.get_rigid_body("carcass"),
        "drawers articulation is rooted on the carcass",
    )

    poses = ctx.sample_joint("drawer_slide_r2c2", positions=(0.0, 0.13, TRAVEL))
    path = ctx.track_point("drawer_r2c2", (column_x(2), FRONT_FRONT, row_z(2)), poses)
    drift = max(abs(p[1] - path[0][1] - pos)
                for p, pos in zip(path, (0.0, 0.13, TRAVEL)))
    lateral = max(abs(p[0] - path[0][0]) + abs(p[2] - path[0][2]) for p in path)
    ctx.check(
        "drawer_travels_straight_out",
        drift < 1e-6 and lateral < 1e-6,
        f"handle point moved +Y by {TRAVEL} m; drift {drift:.2e}, lateral {lateral:.2e}",
    )
    ctx.expect_contact_at_poses(
        "drawer_r2c2", "carcass", poses,
        shape_a="bottom", shape_b="runner_r2c2_left",
        name="drawer_stays_on_its_runner_while_sliding",
    )
    ctx.expect_within_at_poses(
        "drawer_r2c2", "carcass", poses, axes="xz", margin=0.001,
        name="drawer_box_stays_in_its_opening",
    )
    for neighbour in ("drawer_r1c2", "drawer_r2c1", "drawer_r2c3"):
        ctx.expect_no_collision_at_poses(
            "drawer_r2c2", neighbour, poses, name=f"clear_of_{neighbour}_while_sliding"
        )

    # ------------------------------------------- seating, guidance and fit-up
    for row in (1, 2, 3):
        for col in (1, 2, 3):
            drawer = f"drawer_r{row}c{col}"
            tag = f"r{row}c{col}"
            for side in ("left", "right"):
                ctx.expect_contact(
                    drawer, "carcass",
                    shape_a="bottom", shape_b=f"runner_{tag}_{side}",
                    name=f"{drawer}_rides_on_{side}_runner",
                )
                # The box side slides past the runner that guides it.
                if side == "left":
                    positive, positive_shape = drawer, f"side_{side}"
                    negative, negative_shape = "carcass", f"runner_{tag}_{side}"
                else:
                    positive, positive_shape = "carcass", f"runner_{tag}_{side}"
                    negative, negative_shape = drawer, f"side_{side}"
                ctx.expect_gap(
                    positive, negative, axis="x",
                    positive_shape=positive_shape, negative_shape=negative_shape,
                    min_gap=0.0007, max_gap=0.0016,
                    name=f"{drawer}_{side}_slide_clearance",
                )
            ctx.expect_within(
                drawer, "carcass", axes="xz", margin=0.002,
                name=f"{drawer}_seated_in_its_opening",
            )

    # Overlay fronts clear the carcass frame and the neighbouring fronts.
    ctx.expect_gap(
        "drawer_r2c2", "carcass", axis="y",
        positive_shape="front_panel", negative_shape="mid_rail_2",
        min_gap=0.001, max_gap=0.0025, name="front_panel_clears_the_frame",
    )
    ctx.expect_gap(
        "drawer_r2c2", "drawer_r2c1", axis="x",
        positive_shape="front_panel", negative_shape="front_panel",
        min_gap=0.005, max_gap=0.007, name="neighbouring_front_gap",
    )
    ctx.expect_gap(
        "drawer_r2c2", "drawer_r1c2", axis="z",
        positive_shape="front_panel", negative_shape="front_panel",
        min_gap=0.005, max_gap=0.007, name="stacked_front_gap",
    )

    # Hardware and reeding stand proud of the painted front face.
    face = ctx.shape_world_bounds("drawer_r2c2", "front_panel")[1]
    crest = ctx.shape_world_bounds("drawer_r2c2", "flutes")[1]
    ctx.expect_metric(
        "flute_crest_proud_of_front", round(crest[1] - face[1], 4),
        minimum=0.003, maximum=0.005, unit="m",
        details="reeded field stands clear of the flat front border",
    )
    bracket = ctx.shape_world_bounds("drawer_r2c2", "corner_bracket_x1_z1")[1]
    ctx.expect_metric(
        "corner_bracket_proud_of_front", round(bracket[1] - face[1], 4),
        minimum=0.0015, maximum=0.0035, unit="m",
        details="brass corner brackets are surface-mounted",
    )
    ctx.expect_gap(
        "drawer_r2c2", "drawer_r2c2", axis="y",
        positive_shape="handle_bar", negative_shape="front_panel",
        min_gap=0.018, name="brass_bar_handle_stands_proud",
    )
    flutes = object_model.get_rigid_body("drawer_r2c2").shape("flutes")
    crests = sum(
        1 for f in flutes.faces()
        if f.normal_at().Y > 0.9 and 4e-4 < f.area < 9e-4
    )
    ctx.expect_metric("reeds_per_drawer_front", crests, minimum=25, unit="reed")

    # ------------------------------------------------- real solid construction
    carcass_volume = ctx.measure_geometry("carcass").signed_volume
    drawer_volume = ctx.measure_geometry("drawer_r2c2").signed_volume
    ctx.record_metric("carcass_wood_volume", round(carcass_volume, 4), unit="m^3")
    ctx.record_metric("drawer_wood_volume", round(drawer_volume, 5), unit="m^3")
    ctx.expect_positive_volume("carcass", minimum=0.02)
    ctx.expect_positive_volume("drawer_r2c2", minimum=0.002)
    ctx.check(
        "drawer_is_a_box_not_a_block",
        drawer_volume < 0.4 * FRONT_W * FRONT_H * (Y_FRONT - BOX_BACK_Y0),
        f"drawer solids {drawer_volume:.5f} m^3, so the box is hollow",
    )

    for png, caption in (
        ("qa/previews/overall.png", "three-quarter view of the finished chest"),
        ("qa/previews/front.png", "three-by-three grid of fluted drawer fronts"),
        ("qa/previews/drawers_open.png", "three drawers slid out of the carcass"),
        ("qa/previews/slide_strip.png", "one drawer sampled through its slide"),
        ("qa/previews/section.png", "cross-section through the runners and drawer boxes"),
        ("qa/previews/flute_closeup.png", "reeded field, bar handle and corner bracket"),
    ):
        ctx.attach_artifact(png, name=png.split("/")[-1], caption=caption)
    return ctx.report()
