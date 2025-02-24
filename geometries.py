from mpi4py import MPI
import gmsh
from dolfinx.io import gmshio
import dolfinx.mesh as msh
import numpy as np

def cubic_domain(side_box = 0.11, radius = 0.1, lc = 8e-3, model_name = "Cubic"):

    gmsh.initialize()
    comm = MPI.COMM_WORLD
    model_rank = 0
    model = gmsh.model()
    gmsh.model.add(model_name)

    # Definition of the points
    p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
    p2 = gmsh.model.geo.addPoint(side_box, 0, 0, lc)
    p3 = gmsh.model.geo.addPoint(side_box, side_box, 0, lc)
    p4 = gmsh.model.geo.addPoint(0, side_box, 0, lc)

    p5 = gmsh.model.geo.addPoint(0, 0, side_box, lc)
    p6 = gmsh.model.geo.addPoint(side_box, 0, side_box, lc)
    p7 = gmsh.model.geo.addPoint(side_box, side_box, side_box, lc)
    p8 = gmsh.model.geo.addPoint(0, side_box, side_box, lc)

    p9 = gmsh.model.geo.addPoint(side_box, radius, 0, lc)
    p10 = gmsh.model.geo.addPoint(side_box, 0, radius, lc)

    # Definition of the lines
    l1 = gmsh.model.geo.addLine(p1, p4)
    l2 = gmsh.model.geo.addLine(p4, p3)
    l3 = gmsh.model.geo.addLine(p3, p9)
    l4 = gmsh.model.geo.addLine(p9, p2)
    l5 = gmsh.model.geo.addLine(p2, p1)

    l6 = gmsh.model.geo.addLine(p5, p6)
    l7 = gmsh.model.geo.addLine(p6, p7)
    l8 = gmsh.model.geo.addLine(p7, p8)
    l9 = gmsh.model.geo.addLine(p8, p5)

    l10 = gmsh.model.geo.addLine(p2, p10)
    l11 = gmsh.model.geo.addLine(p10, p6)
    l12 = gmsh.model.geo.addLine(p3, p7)
    l13 = gmsh.model.geo.addLine(p4, p8)
    l14 = gmsh.model.geo.addLine(p1, p5)

    # definition of the quarter circle
    c15 = gmsh.model.geo.addCircleArc(p9, p2, p10)

    # Curve loops
    cl1 = [l1, l2, l3, l4, l5]
    cl2 = [l6, l7, l8, l9]
    cl3 = [-l3, l12, -l7, -l11, -c15]
    cl4 = [c15, -l10, -l4]
    cl5 = [-l2, l13, -l8, -l12]
    cl6 = [-l5, l10, l11, -l6, -l14]
    cl7 = [-l1, l14, -l9, -l13]

    # Surfaces
    s1 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl1)])
    s2 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl2)])
    s3 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl3)])
    s4 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl4)])
    s5 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl5)])
    s6 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl6)])
    s7 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl7)])

    # Surface loop
    sl1 = gmsh.model.geo.addSurfaceLoop([s1, s2, s5, s3, s4, s6, s7])

    # Definition of the volume
    v1 = gmsh.model.geo.addVolume([sl1])

    gmsh.model.geo.synchronize()

    # Definition of the physical identities
    gmsh.model.addPhysicalGroup(2, [s4], tag=1)  # Neumann
    gmsh.model.addPhysicalGroup(2, [s7, s5, s2], tag=3)
    gmsh.model.addPhysicalGroup(3, [v1], tag=1)  # Volume physique

    # Generation of the 3D mesh
    gmsh.model.mesh.generate(3)

    # Save the mesh
    gmsh.write(model_name + ".msh")

    # Affect the mesh, volumic tags and surfacic tag to variables
    final_mesh, cell_tags, facet_tags = gmshio.model_to_mesh(model, comm, model_rank)

    # Close gmsh
    gmsh.finalize()
    
    tdim = final_mesh.topology.dim
    fdim = tdim - 1

    xref = [side_box, 0, 0]

    mesh_info = [final_mesh, cell_tags, facet_tags, xref]

    submesh, entity_map = msh.create_submesh(final_mesh, fdim, facet_tags.find(3))[0:2]

    tdim = final_mesh.topology.dim
    fdim = tdim - 1
    submesh_tdim = submesh.topology.dim
    submesh_fdim = submesh_tdim - 1

    # Create the entity maps and an integration measure
    mesh_facet_imap = final_mesh.topology.index_map(fdim)
    mesh_num_facets = mesh_facet_imap.size_local + mesh_facet_imap.num_ghosts
    extract = np.array([entity_map.tolist().index(entity) if entity in entity_map else -1 for entity in range(mesh_num_facets)])
    entity_maps_mesh = {submesh: extract}
    

    mesh_info = [final_mesh, cell_tags, facet_tags, xref]
    submesh_info = [submesh, entity_maps_mesh]

    return mesh_info, submesh_info

def curvedcubic_domain(side_box = 0.11, radius = 0.1, lc = 8e-3, model_name = "rounded_cubic"):

    rc = side_box /10

    gmsh.initialize()
    comm = MPI.COMM_WORLD
    model_rank = 0
    model = gmsh.model()
    gmsh.model.add(model_name)

    # Definition of the points
    p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
    p2 = gmsh.model.geo.addPoint(side_box + rc, 0, 0, lc)
    p3 = gmsh.model.geo.addPoint(side_box + rc, side_box, 0, lc)
    p4 = gmsh.model.geo.addPoint(side_box, side_box + rc, 0, lc)
    p5 = gmsh.model.geo.addPoint(0, side_box + rc, 0, lc)
    p6 = gmsh.model.geo.addPoint(0, 0, side_box + rc, lc)
    p7 = gmsh.model.geo.addPoint(side_box, 0, side_box + rc, lc)
    p8 = gmsh.model.geo.addPoint(side_box, side_box, side_box + rc, lc)
    p9 = gmsh.model.geo.addPoint(0, side_box, side_box + rc, lc)
    p10 = gmsh.model.geo.addPoint(side_box + rc, 0, side_box, lc)
    p11 = gmsh.model.geo.addPoint(side_box + rc, side_box, side_box, lc)
    p12 = gmsh.model.geo.addPoint(side_box, side_box + rc, side_box, lc)
    p13 = gmsh.model.geo.addPoint(0, side_box + rc, side_box, lc)
    p14 = gmsh.model.geo.addPoint(radius, 0, 0, lc)
    p15 = gmsh.model.geo.addPoint(0, 0, radius, lc)

    # Defining points for circles
    pc4 = gmsh.model.geo.addPoint(side_box, side_box, 0, lc)
    pc20_14_23 = gmsh.model.geo.addPoint(side_box, side_box, side_box, lc)
    pc18 = gmsh.model.geo.addPoint(0, side_box, side_box, lc)
    pc10 = gmsh.model.geo.addPoint(side_box, 0, side_box, lc)

    
    # Definition of the lines
    c0 = gmsh.model.geo.addCircleArc(p14, p1, p15)
    l1 = gmsh.model.geo.addLine(p1, p14)
    l2 = gmsh.model.geo.addLine(p14, p2)
    l3 = gmsh.model.geo.addLine(p2, p3)
    c4 = gmsh.model.geo.addCircleArc(p3, pc4, p4)
    l5 = gmsh.model.geo.addLine(p4, p5)
    l6 = gmsh.model.geo.addLine(p5, p1)
    l7 = gmsh.model.geo.addLine(p1, p15)
    l8 = gmsh.model.geo.addLine(p15, p6)
    l9 = gmsh.model.geo.addLine(p6, p7)
    c10 = gmsh.model.geo.addCircleArc(p7, pc10, p10)
    l11 = gmsh.model.geo.addLine(p10, p2)
    l12 = gmsh.model.geo.addLine(p3, p11)
    l13 = gmsh.model.geo.addLine(p11, p10)
    c14 = gmsh.model.geo.addCircleArc(p11, pc20_14_23, p8)
    l15 = gmsh.model.geo.addLine(p7, p8)
    l16 = gmsh.model.geo.addLine(p8, p9)
    l17 = gmsh.model.geo.addLine(p9, p6)
    c18 = gmsh.model.geo.addCircleArc(p9, pc18, p13)
    l19 = gmsh.model.geo.addLine(p13, p12)
    c20 = gmsh.model.geo.addCircleArc(p12, pc20_14_23, p11)
    l21 = gmsh.model.geo.addLine(p12, p4)
    l22 = gmsh.model.geo.addLine(p5, p13)
    c23 = gmsh.model.geo.addCircleArc(p8, pc20_14_23, p12)

    # Curve loops
    cl1 = [l1, c0, -l7]
    cl2 = [l2, -l11, -c10, -l9, -l8, -c0]
    cl3 = [l3, l12, l13, l11]
    cl4 = [c4, -l21, c20, -l12]
    cl5 = [l5, l22, l19, l21]
    cl6 = [l8, -l17, c18, -l22, l6, l7]
    cl7 = [c23, -l19, -c18, -l16]
    cl8 = [l17, l9, l15, l16]
    cl9 = [c14, -l15, c10, -l13]
    cl10 = [-c20, -c23, -c14]
    cl11 = [-l2, -l1, -l6, -l5, -c4, -l3]

    # Surfaces
    s1 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl1)])
    s2 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl2)])
    s3 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl3)])
    s4 = gmsh.model.geo.addSurfaceFilling([gmsh.model.geo.addCurveLoop(cl4)])
    s5 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl5)])
    s6 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl6)])
    s7 = gmsh.model.geo.addSurfaceFilling([gmsh.model.geo.addCurveLoop(cl7)])
    s8 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl8)])
    s9 = gmsh.model.geo.addSurfaceFilling([gmsh.model.geo.addCurveLoop(cl9)])
    s10 = gmsh.model.geo.addSurfaceFilling([gmsh.model.geo.addCurveLoop(cl10)])
    s11 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl11)])

    # Surface loop
    sl1 = gmsh.model.geo.addSurfaceLoop([s1, s2, s5, s3, s4, s6, s7, s8, s9, s10, s11])

    # Definition of the volume
    v1 = gmsh.model.geo.addVolume([sl1])

    gmsh.model.geo.synchronize()

    # Definition of the physical identities
    gmsh.model.addPhysicalGroup(2, [s1], tag=1)  # Neumann
    gmsh.model.addPhysicalGroup(2, [s3, s4, s5, s7, s10, s9, s8], tag=3)
    gmsh.model.addPhysicalGroup(3, [v1], tag=1)  # Volume physique

    # Generation of the 3D mesh
    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.option.setNumber("Mesh.HighOrderOptimize", 2)
    gmsh.model.mesh.generate(3)

    # Save the mesh
    gmsh.write(model_name + ".msh")

    # Affect the mesh, volumic tags and surfacic tag to variables
    final_mesh, cell_tags, facet_tags = gmshio.model_to_mesh(model, comm, model_rank)

    # Close gmsh
    gmsh.finalize()
    
    tdim = final_mesh.topology.dim
    fdim = tdim - 1

    xref = [0, 0, 0]

    mesh_info = [final_mesh, cell_tags, facet_tags, xref]

    submesh, entity_map = msh.create_submesh(final_mesh, fdim, facet_tags.find(3))[0:2]

    tdim = final_mesh.topology.dim
    fdim = tdim - 1
    submesh_tdim = submesh.topology.dim
    submesh_fdim = submesh_tdim - 1

    # Create the entity maps and an integration measure
    mesh_facet_imap = final_mesh.topology.index_map(fdim)
    mesh_num_facets = mesh_facet_imap.size_local + mesh_facet_imap.num_ghosts
    extract = np.array([entity_map.tolist().index(entity) if entity in entity_map else -1 for entity in range(mesh_num_facets)])
    entity_maps_mesh = {submesh: extract}
    

    mesh_info = [final_mesh, cell_tags, facet_tags, xref]
    submesh_info = [submesh, entity_maps_mesh]

    return mesh_info, submesh_info

def spherical_domain(side_box = 0.11, radius = 0.1, lc = 8e-3, model_name = "spherical"):

    gmsh.initialize()
    comm = MPI.COMM_WORLD
    model_rank = 0
    model = gmsh.model()
    gmsh.model.add(model_name)

    # Definition of the points
    p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
    p2 = gmsh.model.geo.addPoint(0, 0, side_box, lc)
    p3 = gmsh.model.geo.addPoint(0, side_box, 0, lc)
    p4 = gmsh.model.geo.addPoint(side_box, 0, 0, lc)

    p5 = gmsh.model.geo.addPoint(0, 0, radius, lc)
    p6 = gmsh.model.geo.addPoint(0, radius, 0, lc)


    # Definition of the lines
    l1 = gmsh.model.geo.addLine(p1, p5)
    l2 = gmsh.model.geo.addLine(p5, p2)
    c3 = gmsh.model.geo.addCircleArc(p2, p1, p3)
    l4 = gmsh.model.geo.addLine(p3, p6)
    l5 = gmsh.model.geo.addLine(p6, p1)

    c6 = gmsh.model.geo.addCircleArc(p5, p1, p6)

    c7 = gmsh.model.geo.addCircleArc(p2, p1, p4)
    l8 = gmsh.model.geo.addLine(p4, p1)

    c9 = gmsh.model.geo.addCircleArc(p4, p1, p3)


    # Curve loops
    cl1 = gmsh.model.geo.addCurveLoop([l1, c6, l5])
    cl2 = gmsh.model.geo.addCurveLoop([l2, c3, l4, -c6])
    cl3 = gmsh.model.geo.addCurveLoop([-l1, -l8, -c7, -l2])
    cl4 = gmsh.model.geo.addCurveLoop([-c9, l8, -l5, -l4])
    cl5 = gmsh.model.geo.addCurveLoop([c9, -c3, c7])


    s1 = gmsh.model.geo.addPlaneSurface([cl1])
    s2 = gmsh.model.geo.addPlaneSurface([cl2])
    s3 = gmsh.model.geo.addPlaneSurface([cl3])
    s4 = gmsh.model.geo.addPlaneSurface([cl4])
    s5 = gmsh.model.geo.addSurfaceFilling([cl5])

    # Surface loop
    sl1 = gmsh.model.geo.addSurfaceLoop([s1, s2, s5, s4, s3])

    # Definition of the volume
    v1 = gmsh.model.geo.addVolume([sl1])

    gmsh.model.geo.synchronize()

    # Definition of the physical identities
    gmsh.model.addPhysicalGroup(2, [s1], tag=1)  # Neumann
    gmsh.model.addPhysicalGroup(2, [s5], tag=3)  # Radiation_r
    gmsh.model.addPhysicalGroup(3, [v1], tag=1)  # Volume physique

    # Generation of the 3D mesh
    gmsh.model.mesh.generate(3)

    # Save the mesh
    gmsh.write(model_name + ".msh")

    # Affect the mesh, volumic tags and surfacic tag to variables
    final_mesh, cell_tags, facet_tags = gmshio.model_to_mesh(model, comm, model_rank)

    # Close gmsh
    gmsh.finalize()
    
    tdim = final_mesh.topology.dim
    fdim = tdim - 1

    xref = [0, 0, 0]

    mesh_info = [final_mesh, cell_tags, facet_tags, xref]

    submesh, entity_map = msh.create_submesh(final_mesh, fdim, facet_tags.find(3))[0:2]

    tdim = final_mesh.topology.dim
    fdim = tdim - 1
    submesh_tdim = submesh.topology.dim
    submesh_fdim = submesh_tdim - 1

    # Create the entity maps and an integration measure
    mesh_facet_imap = final_mesh.topology.index_map(fdim)
    mesh_num_facets = mesh_facet_imap.size_local + mesh_facet_imap.num_ghosts
    extract = np.array([entity_map.tolist().index(entity) if entity in entity_map else -1 for entity in range(mesh_num_facets)])
    entity_maps_mesh = {submesh: extract}

    mesh_info = [final_mesh, cell_tags, facet_tags, xref]
    submesh_info = [submesh, entity_maps_mesh]

    return mesh_info, submesh_info

def ellipsoidal_domain(side_box=0.11, radius=0.1, lc=8e-3, model_name="ellipsoid"):
    gmsh.initialize()
    comm = MPI.COMM_WORLD
    model_rank = 0
    model = gmsh.model()
    gmsh.model.add(model_name)

    # Definition of the points
    p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
    p2 = gmsh.model.geo.addPoint(0, 0, side_box, lc)
    p3 = gmsh.model.geo.addPoint(0, side_box, 0, lc)
    p4 = gmsh.model.geo.addPoint(2*side_box, 0, 0, lc)
    p5 = gmsh.model.geo.addPoint(0, 0, radius, lc)
    p6 = gmsh.model.geo.addPoint(0, radius, 0, lc)

    # Definition of the lines and arcs
    l1 = gmsh.model.geo.addLine(p1, p5)
    l2 = gmsh.model.geo.addLine(p5, p2)
    c3 = gmsh.model.geo.addCircleArc(p2, p1, p3)  # Part of cube's edge
    l4 = gmsh.model.geo.addLine(p3, p6)
    l5 = gmsh.model.geo.addLine(p6, p1)
    c6 = gmsh.model.geo.addCircleArc(p5, p1, p6)  # Circular arc for surface 1

    # Elliptical arcs replacing spherical arcs
    # Elliptical arc from p2 (0,0,side_box) to p4 (side_box,0,0) in x-z plane
    e7 = gmsh.model.geo.addEllipseArc(p2, p1, p4, p4)
    # Elliptical arc from p4 (side_box,0,0) to p3 (0,side_box,0) in x-y plane
    e9 = gmsh.model.geo.addEllipseArc(p4, p1, p4, p3)
    l8 = gmsh.model.geo.addLine(p4, p1)

    # Curve loops
    cl1 = gmsh.model.geo.addCurveLoop([l1, c6, l5])
    cl2 = gmsh.model.geo.addCurveLoop([l2, c3, l4, -c6])
    cl3 = gmsh.model.geo.addCurveLoop([-l1, -l8, -e7, -l2])
    cl4 = gmsh.model.geo.addCurveLoop([-e9, l8, -l5, -l4])
    cl5 = gmsh.model.geo.addCurveLoop([e9, -c3, e7])

    # Surfaces
    s1 = gmsh.model.geo.addPlaneSurface([cl1])
    s2 = gmsh.model.geo.addPlaneSurface([cl2])
    s3 = gmsh.model.geo.addPlaneSurface([cl3])
    s4 = gmsh.model.geo.addPlaneSurface([cl4])
    s5 = gmsh.model.geo.addSurfaceFilling([cl5])  # Ellipsoidal surface

    # Surface loop and volume
    sl1 = gmsh.model.geo.addSurfaceLoop([s1, s2, s5, s4, s3])
    v1 = gmsh.model.geo.addVolume([sl1])

    gmsh.model.geo.synchronize()

    # Physical groups
    gmsh.model.addPhysicalGroup(2, [s1], tag=1)  # Surface quart de disque
    gmsh.model.addPhysicalGroup(2, [s5], tag=3)  # Surface ellipsoïdale
    gmsh.model.addPhysicalGroup(3, [v1], tag=1)   # Volume

    # Mesh generation
    gmsh.model.mesh.generate(3)
    gmsh.write(model_name + ".msh")

    # Extract mesh and tags
    final_mesh, cell_tags, facet_tags = gmshio.model_to_mesh(model, comm, model_rank)
    gmsh.finalize()

    # Remaining code for submesh and entity maps...
    tdim = final_mesh.topology.dim
    fdim = tdim - 1
    xref = [0, 0, 0]
    mesh_info = [final_mesh, cell_tags, facet_tags, xref]

    submesh, entity_map = msh.create_submesh(final_mesh, fdim, facet_tags.find(3))[0:2]
    mesh_facet_imap = final_mesh.topology.index_map(fdim)
    mesh_num_facets = mesh_facet_imap.size_local + mesh_facet_imap.num_ghosts
    extract = np.array([entity_map.tolist().index(entity) if entity in entity_map else -1 for entity in range(mesh_num_facets)])
    entity_maps_mesh = {submesh: extract}
    submesh_info = [submesh, entity_maps_mesh]

    return mesh_info, submesh_info




def half_cubic_domain(side_box = 0.11, radius = 0.1, lc = 8e-3, model_name = "half_cubic"):

    gmsh.initialize()
    comm = MPI.COMM_WORLD
    model_rank = 0
    model = gmsh.model()
    gmsh.model.add(model_name)

    # Definition of the points
    p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
    p2 = gmsh.model.geo.addPoint(side_box/2, 0, 0, lc)
    p3 = gmsh.model.geo.addPoint(side_box/2, side_box, 0, lc)
    p4 = gmsh.model.geo.addPoint(0, side_box, 0, lc)

    p5 = gmsh.model.geo.addPoint(0, 0, side_box, lc)
    p6 = gmsh.model.geo.addPoint(side_box/2, 0, side_box, lc)
    p7 = gmsh.model.geo.addPoint(side_box/2, side_box, side_box, lc)
    p8 = gmsh.model.geo.addPoint(0, side_box, side_box, lc)

    p9 = gmsh.model.geo.addPoint(side_box/2, radius, 0, lc)
    p10 = gmsh.model.geo.addPoint(side_box/2, 0, radius, lc)

    # Definition of the lines
    l1 = gmsh.model.geo.addLine(p1, p4)
    l2 = gmsh.model.geo.addLine(p4, p3)
    l3 = gmsh.model.geo.addLine(p3, p9)
    l4 = gmsh.model.geo.addLine(p9, p2)
    l5 = gmsh.model.geo.addLine(p2, p1)

    l6 = gmsh.model.geo.addLine(p5, p6)
    l7 = gmsh.model.geo.addLine(p6, p7)
    l8 = gmsh.model.geo.addLine(p7, p8)
    l9 = gmsh.model.geo.addLine(p8, p5)

    l10 = gmsh.model.geo.addLine(p2, p10)
    l11 = gmsh.model.geo.addLine(p10, p6)
    l12 = gmsh.model.geo.addLine(p3, p7)
    l13 = gmsh.model.geo.addLine(p4, p8)
    l14 = gmsh.model.geo.addLine(p1, p5)

    # definition of the quarter circle
    c15 = gmsh.model.geo.addCircleArc(p9, p2, p10)

    # Curve loops
    cl1 = [l1, l2, l3, l4, l5]
    cl2 = [l6, l7, l8, l9]
    cl3 = [-l3, l12, -l7, -l11, -c15]
    cl4 = [c15, -l10, -l4]
    cl5 = [-l2, l13, -l8, -l12]
    cl6 = [-l5, l10, l11, -l6, -l14]
    cl7 = [-l1, l14, -l9, -l13]

    # Surfaces
    s1 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl1)])
    s2 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl2)])
    s3 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl3)])
    s4 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl4)])
    s5 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl5)])
    s6 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl6)])
    s7 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl7)])

    # Surface loop
    sl1 = gmsh.model.geo.addSurfaceLoop([s1, s2, s5, s3, s4, s6, s7])

    # Definition of the volume
    v1 = gmsh.model.geo.addVolume([sl1])

    gmsh.model.geo.synchronize()

    # Definition of the physical identities
    gmsh.model.addPhysicalGroup(2, [s4], tag=1)  # Neumann
    gmsh.model.addPhysicalGroup(2, [s7, s5, s2], tag=3)
    gmsh.model.addPhysicalGroup(3, [v1], tag=1)  # Volume physique

    # Generation of the 3D mesh
    gmsh.model.mesh.generate(3)

    # Save the mesh
    gmsh.write(model_name + ".msh")

    # Affect the mesh, volumic tags and surfacic tag to variables
    final_mesh, cell_tags, facet_tags = gmshio.model_to_mesh(model, comm, model_rank)

    # Close gmsh
    gmsh.finalize()
    
    tdim = final_mesh.topology.dim
    fdim = tdim - 1

    xref = [side_box/2, 0, 0]

    mesh_info = [final_mesh, cell_tags, facet_tags, xref]

    submesh, entity_map = msh.create_submesh(final_mesh, fdim, facet_tags.find(3))[0:2]

    tdim = final_mesh.topology.dim
    fdim = tdim - 1
    submesh_tdim = submesh.topology.dim
    submesh_fdim = submesh_tdim - 1

    # Create the entity maps and an integration measure
    mesh_facet_imap = final_mesh.topology.index_map(fdim)
    mesh_num_facets = mesh_facet_imap.size_local + mesh_facet_imap.num_ghosts
    extract = np.array([entity_map.tolist().index(entity) if entity in entity_map else -1 for entity in range(mesh_num_facets)])
    entity_maps_mesh = {submesh: extract}
    

    mesh_info = [final_mesh, cell_tags, facet_tags, xref]
    submesh_info = [submesh, entity_maps_mesh]

    return mesh_info, submesh_info

def broken_cubic_domain(side_box = 0.11, radius = 0.1, lc = 8e-3, model_name = "broken_cubic"):
    #Make an eighth of the cubic domain of a sound box
    
    gmsh.initialize()
    comm = MPI.COMM_WORLD
    model_rank = 0
    model = gmsh.model()
    model_name = "example"
    gmsh.model.add(model_name)

    # Definition of the points
    p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
    p2 = gmsh.model.geo.addPoint(side_box, 0, 0, lc)
    p3 = gmsh.model.geo.addPoint(side_box, side_box, 0, lc)
    p4 = gmsh.model.geo.addPoint(0, side_box, 0, lc)

    p5 = gmsh.model.geo.addPoint(0, 0, side_box, lc)
    p6 = gmsh.model.geo.addPoint(side_box, 0, side_box, lc)
    p7 = gmsh.model.geo.addPoint(side_box, side_box, side_box, lc)
    p8 = gmsh.model.geo.addPoint(0, side_box, side_box, lc)

    p9 = gmsh.model.geo.addPoint(side_box, radius, 0, lc)
    p10 = gmsh.model.geo.addPoint(side_box, 0, radius, lc)

    p11 = gmsh.model.geo.addPoint(side_box/2, side_box, side_box/2, lc)
    p12 = gmsh.model.geo.addPoint(side_box/2, 0, side_box/2, lc)

    # Definition of the lines
    l1 = gmsh.model.geo.addLine(p1, p4)
    l2 = gmsh.model.geo.addLine(p4, p3)
    l3 = gmsh.model.geo.addLine(p3, p9)
    l4 = gmsh.model.geo.addLine(p9, p2)
    l5 = gmsh.model.geo.addLine(p2, p1)

    l6 = gmsh.model.geo.addLine(p5, p6)
    l7 = gmsh.model.geo.addLine(p6, p7)
    l8 = gmsh.model.geo.addLine(p7, p8)
    l9 = gmsh.model.geo.addLine(p8, p5)

    l10 = gmsh.model.geo.addLine(p2, p10)
    l11 = gmsh.model.geo.addLine(p10, p6)
    l12 = gmsh.model.geo.addLine(p3, p7)
    l13 = gmsh.model.geo.addLine(p4, p11)
    l14 = gmsh.model.geo.addLine(p11, p8)
    l16 = gmsh.model.geo.addLine(p1, p12)
    l17 = gmsh.model.geo.addLine(p12, p5)
    l18 = gmsh.model.geo.addLine(p11, p12)
    # definition of the quarter circle
    c15 = gmsh.model.geo.addCircleArc(p9, p2, p10)

    # Curve loops
    cl1 = [l1, l2, l3, l4, l5]
    cl2 = [l6, l7, l8, l9]
    cl3 = [-l3, l12, -l7, -l11, -c15]
    cl4 = [c15, -l10, -l4]
    cl5 = [-l2, l13, l14, -l8, -l12]
    cl6 = [-l5, l10, l11, -l6, -l17, -l16]
    cl7 = [-l1, l16, -l18, -l13]
    cl8 = [l18, l17, -l9, -l14]

    # Surfaces
    s1 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl1)])
    s2 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl2)])
    s3 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl3)])
    s4 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl4)])
    s5 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl5)])
    s6 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl6)])
    s7 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl7)])
    s8 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl8)])
    # Surface loop
    sl1 = gmsh.model.geo.addSurfaceLoop([s1, s2, s5, s3, s4, s6, s7, s8])

    # Definition of the volume
    v1 = gmsh.model.geo.addVolume([sl1])

    gmsh.model.geo.synchronize()

    # Definition of the physical identities
    gmsh.model.addPhysicalGroup(2, [s4], tag=1)  # Neumann
    gmsh.model.addPhysicalGroup(2, [s8, s7, s5, s2], tag=3)
    gmsh.model.addPhysicalGroup(3, [v1], tag=1)  # Volume physique

    # Generation of the 3D mesh
    gmsh.model.mesh.generate(3)

    # Save the mesh
    gmsh.write("Broken_cubic.msh")

    # Affect the mesh, volumic tags and surfacic tag to variables
    final_mesh, cell_tags, facet_tags = gmshio.model_to_mesh(model, comm, model_rank)

    # Close gmsh
    gmsh.finalize()
    
    tdim = final_mesh.topology.dim
    fdim = tdim - 1

    xref = [side_box, 0, 0]

    mesh_info = [final_mesh, cell_tags, facet_tags, xref]

    submesh, entity_map = msh.create_submesh(final_mesh, fdim, facet_tags.find(3))[0:2]

    tdim = final_mesh.topology.dim
    fdim = tdim - 1
    submesh_tdim = submesh.topology.dim
    submesh_fdim = submesh_tdim - 1

    # Create the entity maps and an integration measure
    mesh_facet_imap = final_mesh.topology.index_map(fdim)
    mesh_num_facets = mesh_facet_imap.size_local + mesh_facet_imap.num_ghosts
    extract = np.array([entity_map.tolist().index(entity) if entity in entity_map else -1 for entity in range(mesh_num_facets)])
    entity_maps_mesh = {submesh: extract}

    mesh_info = [final_mesh, cell_tags, facet_tags, xref]
    submesh_info = [submesh, entity_maps_mesh]

    return mesh_info, submesh_info