# scenario

Zone roles, the furnished occupancy grid, and the four base scenarios `evaluation generate populate` writes. Part of [generation](../README.md).

| File | Purpose |
|---|---|
| `zones.py` | zone name -> `Role` by substring rules, then the reviewed `zone_roles.yaml` |
| `zone_roles.yaml` | reviewed name -> role mapping (`source: human` rows are never overwritten) |
| `occupancy.py` | each world object's footprint, from `Obstacle.footprint()` (the asset's annotated bounds through arena_simulation_setup), else the 2 m box `WorldManager` stamps for an unannotated one |
| `zone_screen.py` | the runtime's rendered map (`runtime_map`) with furniture stamped on; which zones may be homes (every admissible cell 1.5 m off the route) |
| `route.py` | the robot's route copied from a shipped scenario, lengthened to a 30 s episode or capped |
| `populate.py` | the bases `normal`, `break`, `night`, `empty`: talker pairs plus typed walkers on multi-room routes |
| `materialize.py` | base writer, `GENERATED_MARKER`, pruning of generated bases |
| `agent_types/talker.yaml` | behaviour-tree type copied beside every base that has talkers |

A base is `PEOPLE_PER_ZONE` people per zone of the profile's role times a per-world crowd factor, at least four: `TALKER_SHARE` of them in pairs, one pair per room, the rest walkers whose routes cover at least 15 m. Walkers are humansim builtin types weighted per profile (`MIX`); one drawn faster than HumanSim's 1.5 m/s default cap is `hurried`, whose own cap is 2.4 m/s. No base has sources or sinks: the `dynamic:` cast is the whole population.

`RUNTIME_SAFE_DIST` (0.35 m) mirrors `task.episode.spawn.obstacle_clearance`, the clearance the runtime validates zone draws at; change both together.
