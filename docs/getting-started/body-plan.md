# Body plan

A body plan describes the kinematic tree QuickIK solves against – a robot's joints, or an animal's skeleton – loaded once from JSON and reused across every solve. Unlike a typical robotics kinematic tree, which separates joints (that move) from frames or links (that get tracked), here every joint doubles as a keypoint: its world position is always available as a potential target, whether or not it carries a rotational DOF of its own. This is what lets one uniform `KeypointObservation` list – one entry per joint, in the body plan's own tree order – describe a whole tracked pose, DOF-bearing joints and fixed leaf keypoints alike.

One modeling consequence worth knowing: a joint's own DOF reorients its *children*, not itself – rotating a joint never moves its own keypoint, only the keypoints downstream of it. So every DOF needs at least one keypoint further down its own chain to be observable at all; a chain that ends exactly at its last DOF-bearing joint, with nothing past it, leaves that DOF's angle undetermined by any observation. This is why even a fixed, 0-DOF "tip" joint (a fingertip, a fly's claw, a robot's end effector) is usually worth keeping in the body plan even though it never actuates anything itself.

By default the root is a free-floating base with its own 6 DOFs (position and rotation), solved for like everything else – this fits a tracked animal or a robot free to move through the world. Setting the top-level `"fixed_base": true` instead anchors the root in place (e.g. a robot arm bolted to a table), removing those 6 DOFs from the state entirely. A *semi*-fixed base – one that only slides along a rail or spins on a turntable – isn't a separate setting: keep `fixed_base` set and give the root a zero-offset child joint carrying that one hinge/slide DOF, then attach the rest of the body to that joint instead of directly to the root. Because a joint's own DOF only moves its descendants (see above), this joint acts as exactly that one-DOF base – and unlike the root's own DOFs, it gets `limits` and `weight_scaler` like any other DOF.

??? note "Body plan JSON schema"
    ```json
    {
      "fixed_base": false,
      "joints": [
        {
          "name": "root",
          "parent": null,
          "offset_pos": [0.0, 0.0, 0.0],
          "offset_quat": [1.0, 0.0, 0.0, 0.0],
          "weight_scaler": 1.0,
          "dofs": []
        },
        {
          "name": "elbow",
          "parent": "root",
          "offset_pos": [1.0, 0.0, 0.0],
          "offset_quat": [1.0, 0.0, 0.0, 0.0],
          "weight_scaler": 1.0,
          "dofs": [
            {
              "axis": [0.0, 0.0, 1.0],
              "type": "hinge",
              "neutral": 0.0,
              "weight_scaler": 1.0,
              "limits": [-3.0, 3.0]
            }
          ]
        },
        {
          "name": "wrist",
          "parent": "elbow",
          "offset_pos": [1.0, 0.0, 0.0],
          "offset_quat": [1.0, 0.0, 0.0, 0.0],
          "weight_scaler": 1.0,
          "dofs": []
        }
      ]
    }
    ```

    - `fixed_base`: whether the root is fixed in the world rather than a free-floating base. Optional, defaults to `false`.
    - `parent`: joint name, or `null` for the root.
    - `offset_pos`/`offset_quat`: this joint's offset from its parent.
    - `weight_scaler`: multiplied together with each frame's `KeypointObservation`'s `weight` for this joint's keypoint. Optional, defaults to `1.0`.
    - `dofs`: this joint's degrees of freedom, each with:
        - `type`: `"hinge"` (rotational) or `"slide"` (translational).
        - `axis`: rotation/translation axis in local frame.
        - `neutral`: neutral angle (radians) or position.
        - `limits`: optional `[min, max]` limits; unbounded if omitted or `null`.
        - `weight_scaler`: multiplied together with `SolverConfig`'s `weight` for this DOF's deviation-from-neutral penalty. Optional, defaults to `1.0`.

    See the [API reference](../api/rust/quickik/body_plan/index.html) for the full schema.
