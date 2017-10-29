
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

/////////////////////////////////////////

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>


#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>


////////////////////////////////////////
// Curl integrator
#include <deal.II/integrators/maxwell.h>

using namespace dealii;

/*
int dim (3);

class ChiInit : public Function<3>
{
public:
  ChiInit () : Function<3> () {}
  virtual double value (const Point<3> &p) const;
};

double ChiInit<3>::value (const Point<3> &p) const
{
      double arg = p[1]*p[2];
      return arg;
}
*/

class R2007
{
public:
  R2007 ();

  void run ();

private:
  void make_grid ();
  void setup_system ();
  void set_IC ();
  void assemble_system ();
  void solve ();
  void output_results () const;

  Triangulation<3>     triangulation;
  FESystem<3>          fe;
  DoFHandler<3>        dof_handler;

// Default variables that we do not use here
  BlockSparsityPattern      sparsity_pattern;
  BlockSparseMatrix<double> system_matrix;

  BlockVector<double>       solution;
  BlockVector<double>       system_rhs;

};


// Initial values for the simple problem described in Reisenegger et al. 2007
// It evolves a pertrubation in poloidal field with
// given initial n_e and torroidal magnetic field.

template <int dim>
class IC_R2007 : public Function<dim>
{
public:
  IC_R2007 () : Function<dim>(dim+1) {}

  // return all components at one point
  void   vector_value (const Point<dim>   &p,
                       Vector<double>     &value) const;

};

template <int dim>
void
IC_R2007<dim>::vector_value (const Point<dim> &p,
                        Vector<double>   &values) const
{
  Assert (values.size() == dim+1,
          ExcDimensionMismatch (values.size(), dim+1));

  // Physical constants
  const double c = 3e10; // cm/s
  const double e = 1; // electron charge

  // Model constatns
  const double n0 = 1.0*1e36; // Central electron density in cm^-3
  const double r0 = 1.0*1e6;  // Star radius 1e6 cm = 10 km
  const double B0 = 1e14;     // Magnitude of the torroidal field in G
  const double B1 = 1e9;     // Magnitude of the poloidal field in G (B0>>B1)
                              // since B1 is a pertrubation

  //  Help variables
  double r;           // Radius
  double Chi, Chi0;
  double r_cyl;       // Cylindrical radius
  double n;           // Electron number density
  double t1,t2,t0;    // Coordinates in cm
  double B;           // Magnetic field in G

  double phi;         // Angular coordinate (cylindrical)

  Point<dim>   px;
  px = p;

  t0 = 1.0 * p[0];
  t1 = 1.0 * p[1];
  t2 = 1.0 * p[2];

  // http://en.cppreference.com/w/cpp/numeric/math/atan2
  // ATAN2 is arctan function that takes into account quadrants
  phi = std::atan2(t1,t2);

  r_cyl = std::sqrt(t1*t1+t2*t2);
  r = std::sqrt(t0*t0+t1*t1+t2*t2);
  n = n0 * (1.0 - r*r/r0/r0);

  Chi = c/4.0/numbers::PI/n/e/(r_cyl*r_cyl+1e3*1e3);
  Chi0 = c/4.0/numbers::PI/n0/e/(r0*r0+1e3*1e3);

  // Chi should always be positive by construction. This check is probably unnecesasry.
  if (Chi<0) Chi=-Chi;

  // Toroidal field magnitude
  B = B0*r0*(Chi0*Chi0/Chi/Chi);

  // return values in order 3 components for B and one for n_e
  values(0) = B1;
  values(1) = B*std::sin(phi+numbers::PI/2.);
  values(2) = B*std::cos(phi+numbers::PI/2.);
  values(3) = n;
}


R2007::R2007 ()
  :
  fe (FESystem<3>(FE_Q<3>(1), 3), 1,
      FE_Q<3>(1),                 1),
  dof_handler (triangulation)
{}


void R2007::make_grid ()
{
  Point<3> center (0,0,0);
  GridGenerator::hyper_shell (triangulation,
                              center, 0.01e6, 0.99e6, 0, true);
  static const SphericalManifold<3> manifold_description(center);
  triangulation.set_all_manifold_ids(0);
  triangulation.set_manifold (0, manifold_description);
  triangulation.refine_global (3);
}

void R2007::setup_system ()
{
  dof_handler.distribute_dofs (fe);

  // Sort dof in order to have a block matrix
  DoFRenumbering::component_wise (dof_handler);

  // Number of DOFs per component
  std::vector<types::global_dof_index> dofs_per_component (4);
  DoFTools::count_dofs_per_component (dof_handler, dofs_per_component);
  const unsigned int n_B = dofs_per_component[0],
                     n_ne = dofs_per_component[3];

    std::cout << "Number of active cells: "
              << triangulation.n_active_cells()
              << std::endl
              << "Total number of cells: "
              << triangulation.n_cells()
              << std::endl
              << "Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << " (" << n_ne << '+' << n_B << ')'
              << std::endl;


// Initialize all matrices and vectors

BlockDynamicSparsityPattern dsp(2, 2);
dsp.block(0, 0).reinit (n_B*3, n_B*3);
dsp.block(1, 0).reinit (n_ne, n_B*3);
dsp.block(0, 1).reinit (n_B*3, n_ne);
dsp.block(1, 1).reinit (n_ne, n_ne);
dsp.collect_sizes ();
DoFTools::make_sparsity_pattern (dof_handler, dsp);
sparsity_pattern.copy_from(dsp);
system_matrix.reinit (sparsity_pattern);

solution.reinit (2);
solution.block(0).reinit (n_B*3);
solution.block(1).reinit (n_ne);
solution.collect_sizes ();


// Not used at this moment
system_rhs.reinit (2);
system_rhs.block(0).reinit (n_B*3);
system_rhs.block(1).reinit (n_ne);
system_rhs.collect_sizes ();

}


void R2007::set_IC ()
{
  ConstraintMatrix constraints;
  constraints.close();

  // Projecting ICs onto the solution vector
  VectorTools::project (dof_handler,
                        constraints,
                        QGauss<3>(3),
                        IC_R2007<3>(),
                        solution);
}

void R2007::assemble_system ()
{

  QGauss<3>  quadrature_formula(2);

  FEValues<3> fe_values (fe, quadrature_formula,
                         update_values | update_gradients | update_JxW_values);

  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();

  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);

  std::cout << "DOF per cells: "
            << dofs_per_cell
            << std::endl
            << "Q points: "
            << n_q_points
            << std::endl;

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  for (const auto &cell: dof_handler.active_cell_iterators())
    {
  //
      fe_values.reinit (cell);

  //
      cell_matrix = 0;
      cell_rhs = 0;
      for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        {
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            for (unsigned int j=0; j<dofs_per_cell; ++j)
              cell_matrix(i,j) += (fe_values.curl (i, q_index) *
                                   fe_values.shape_grad (j, q_index) *
                                   fe_values.JxW (q_index));
            // for (unsigned int i=0; i<dofs_per_cell; ++i)
          //   cell_rhs(i) += (fe_values.shape_value (i, q_index) *
          //                   1 *
          //                   fe_values.JxW (q_index));
        }

      cell->get_dof_indices (local_dof_indices);

      for (unsigned int i=0; i<dofs_per_cell; ++i)
        for (unsigned int j=0; j<dofs_per_cell; ++j)
          system_matrix.add (local_dof_indices[i],
                             local_dof_indices[j],
                             cell_matrix(i,j));

    }
  //
  //     for (unsigned int i=0; i<dofs_per_cell; ++i)
  //       system_rhs(local_dof_indices[i]) += cell_rhs(i);
  //   }
  //
  //
  //
  // std::map<types::global_dof_index,double> boundary_values;
  // VectorTools::interpolate_boundary_values (dof_handler,
  //                                           0,
  //                                           Functions::ZeroFunction<3>(),
  //                                           boundary_values);
  //
  // MatrixTools::apply_boundary_values (boundary_values,
  //                                     system_matrix,
  //                                     solution,
  //                                     system_rhs);
}



void R2007::solve ()
{
  //
  // SolverControl           solver_control (1000, 1e-12);
  //
  // SolverCG<>              solver (solver_control);
  //
  //
  // solver.solve (system_matrix, solution, system_rhs,
  //               PreconditionIdentity());

}



void R2007::output_results () const
{
// Taken from step-20
  std::vector<std::string> solution_names(3, "u");
  solution_names.push_back ("p");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  interpretation (3,
                  DataComponentInterpretation::component_is_part_of_vector);
  interpretation.push_back (DataComponentInterpretation::component_is_scalar);
  DataOut<3> data_out;
  data_out.add_data_vector (dof_handler, solution, solution_names, interpretation);
  data_out.build_patches (1+1);
  std::ofstream output ("solution.vtk");
  data_out.write_vtk (output);

}

void R2007::run ()
{
  make_grid ();
  setup_system ();
  set_IC ();
  assemble_system ();
//  solve ();
  output_results ();
}

int main ()
{
  deallog.depth_console (2);

  R2007 laplace_problem;
  laplace_problem.run ();

  return 0;
}
