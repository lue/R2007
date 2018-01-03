
using namespace dealii;


class R2007
{
public:
  R2007 ();

  void run ();

private:
  void make_grid ();
  void setup_system ();
  void set_IC ();
  void assemble_system (bool test_output);
  void assemble_system_test (bool test_output);
  void propagate ();
  void output_results (const unsigned int timestep_number) const;
  void output_results_test (std::string s) const;

  Triangulation<3>     triangulation;
  FESystem<3>          fe;
  DoFHandler<3>        dof_handler;

// Default variables that we do not use here
  BlockSparsityPattern      sparsity_pattern;
  BlockSparseMatrix<double> system_matrix;

  BlockVector<double>       solution;
  BlockVector<double>       test_solution;
  BlockVector<double>       temp_field;
  BlockVector<double>       temp_count;

  BlockVector<double>       system_rhs;

};

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
                              center, 0.1e6, 0.999e6, 96, true);
  static const SphericalManifold<3> manifold_description(center);
  triangulation.set_all_manifold_ids(0);
  triangulation.set_manifold (0, manifold_description);
  triangulation.refine_global (2);
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

test_solution.reinit (2);
test_solution.block(0).reinit (n_B*3);
test_solution.block(1).reinit (n_ne);
test_solution.collect_sizes ();

temp_field.reinit (2);
temp_field.block(0).reinit (n_B*3);
temp_field.block(1).reinit (n_ne);
temp_field.collect_sizes ();

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

void R2007::assemble_system (bool test_output)
{

  const FEValuesExtractors::Vector Bfield (0);
  const FEValuesExtractors::Scalar ne (3);

  QGauss<3>  quadrature_formula(3);

  FEValues<3> fe_values (fe, quadrature_formula,
                         update_values | update_gradients | update_quadrature_points | update_JxW_values);

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
  std::vector<Vector<double> > local_solution_values (n_q_points,
                                                      Vector<double> (3+1));

  std::vector<Tensor<1,3> > cell_test_solution (dofs_per_cell);

  std::vector<Tensor<1,3> > local_Bfield_values (n_q_points);
  std::vector<Tensor<1,3> > local_vector_values (n_q_points);
  // std::vector<Tensor<1,3> > local_Bfield_curls (n_q_points);
  std::vector<typename dealii::internal::CurlType<3>::type> local_Bfield_curls (n_q_points);
  std::vector<Tensor<2,3> > local_Bfield_gradients (n_q_points);
  std::vector<double> local_ne_values (n_q_points);

  Tensor<1,3> temp1, temp2, temp3;

  temp_field = 0;
  temp_count.reinit(temp_field);
  temp_count = 0;
  double scale = 1e25;


  // Step 1
  // Get v = ∇ x B / n_e
  for (const auto &cell: dof_handler.active_cell_iterators())
    {
      // Reinit local
      fe_values.reinit (cell);
      cell->get_dof_indices (local_dof_indices);
      // Extract all local values
      fe_values.get_function_values (solution,
                                     local_solution_values);
      fe_values[Bfield].get_function_values (solution,
                                     local_Bfield_values);
      // fe_values[Bfield].get_function_gradients (solution,
      //                                local_Bfield_gradients);
      fe_values[Bfield].get_function_curls (solution,
                                     local_Bfield_curls);
      fe_values[ne].get_function_values (solution,
                                     local_ne_values);

      cell_matrix = 0;
      cell_rhs = 0;
      cell_test_solution = local_Bfield_curls;

      for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        {
          for (unsigned int j=3; j<dofs_per_cell; j=j+4)
          {
            temp_field(local_dof_indices[j]) += local_ne_values[q_index] * fe_values.shape_value(j, q_index);
            temp_count(local_dof_indices[j]) += fe_values.shape_value(j, q_index);
          }
          for (unsigned int i=0; i<3; ++i)
            {
              for (unsigned int j=i; j<dofs_per_cell; j=j+4)
              {
                temp_field(local_dof_indices[j]) += scale * local_Bfield_curls[q_index][i] / local_ne_values[q_index]
                                                                    * fe_values.shape_value(j, q_index);
                temp_count(local_dof_indices[j]) += fe_values.shape_value(j, q_index);
              }
            }

        }
    }

    for (unsigned int i=0; i<temp_field.size(); ++i)
    {
      temp_field(i) /= temp_count(i);// / (dofs_per_cell / 4.);
      // TODO: This fix is not appropriate. A better solution should exist
    }

    // Output the results from step 1
    test_solution = (temp_field);
    std::string fname ("curlB_over_ne.vtk");
    if (test_output)
    {
      output_results_test (fname);
    }
    temp_field = 0;


    // Step 2
    // Get v = (∇ x B / n_e) x B

    for (const auto &cell: dof_handler.active_cell_iterators())
      {
        // Reinit local
        fe_values.reinit (cell);
        cell->get_dof_indices (local_dof_indices);
        // Extract all local values
        fe_values[Bfield].get_function_values (solution,
                                       local_Bfield_values);
        fe_values[Bfield].get_function_values (test_solution,
                                       local_vector_values);
        fe_values[ne].get_function_values (test_solution,
                                       local_ne_values);

        // Get v x B
        for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
          {
            for (unsigned int i=0; i<3; ++i)
              {
                temp1[i] = local_vector_values[q_index][i];
                temp2[i] = local_Bfield_values[q_index][i];
              }
            temp3 = cross_product_3d(temp1, temp2);
            for (unsigned int i=0; i<3; ++i)
              {
                for (unsigned int j=i; j<dofs_per_cell; j=j+4)
                {
                  temp_field(local_dof_indices[j]) += temp3[i] * fe_values.shape_value(j, q_index);
                  temp_count(local_dof_indices[j]) += fe_values.shape_value(j, q_index);
                }
              }

            // Copy scalar field
            for (unsigned int j=3; j<dofs_per_cell; j=j+4)
            {
              temp_field(local_dof_indices[j]) += local_ne_values[q_index] * fe_values.shape_value(j, q_index);
              temp_count(local_dof_indices[j]) += fe_values.shape_value(j, q_index);
            }
          }
      }


    for (unsigned int i=0; i<temp_field.size(); ++i)
      {
        temp_field(i) /= temp_count(i);// / (dofs_per_cell / 4.);
        // TODO: This fix is not appropriate. A better solution should exist
      }

    test_solution = (temp_field);
    if (test_output)
    {
      fname = "v_cross_B.vtk";
      output_results_test (fname);
    }
    temp_field = 0;

    // Step 3
    // Now we need another loop to evaluate curl again
    for (const auto &cell: dof_handler.active_cell_iterators())
      {
        // Reinit local
        fe_values.reinit (cell);
        cell->get_dof_indices (local_dof_indices);
        // Extract all local values
        fe_values[Bfield].get_function_curls (test_solution,
                                       local_Bfield_curls);
        fe_values[ne].get_function_values (test_solution,
                                       local_ne_values);


        cell_matrix = 0;
        cell_rhs = 0;

        for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
          {
            for (unsigned int i=0; i<3; ++i)
              {
                for (unsigned int j=i; j<dofs_per_cell; j=j+4)
                {
                  temp_field(local_dof_indices[j]) += local_Bfield_curls[q_index][i] * fe_values.shape_value(j, q_index);
                  temp_count(local_dof_indices[j]) += fe_values.shape_value(j, q_index);
                }
              }
            // Copy scalar field
            for (unsigned int j=3; j<dofs_per_cell; j=j+4)
              {
                // TODO: proper d n_e/dt
                temp_field(local_dof_indices[j]) += 0;//local_ne_values[q_index] * fe_values.shape_value(j, q_index);
                temp_count(local_dof_indices[j]) += fe_values.shape_value(j, q_index);
              }
          }


      }

    for (unsigned int i=0; i<temp_field.size(); ++i)
      {
        temp_field(i) /= temp_count(i);// / (dofs_per_cell / 4.);
        // TODO: This fix is not appropriate. A better solution should exist
      }

    test_solution = (temp_field);
    if (test_output)
    {
      fname = "curl_v_cross_B.vtk";
      output_results_test (fname);
    }
}


void R2007::assemble_system_test (bool test_output)
{
  std::string fname;
  const FEValuesExtractors::Vector Bfield (0);
  const FEValuesExtractors::Scalar ne (3);

  QGauss<3>  quadrature_formula(3);

  FEValues<3> fe_values (fe, quadrature_formula,
                         update_values | update_gradients | update_quadrature_points | update_JxW_values);

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
  std::vector<Vector<double> > local_solution_values (n_q_points,
                                                      Vector<double> (3+1));

  std::vector<Tensor<1,3> > cell_test_solution (dofs_per_cell);

  std::vector<Tensor<1,3> > local_Bfield_values (n_q_points);
  std::vector<Tensor<1,3> > local_vector_values (n_q_points);
  // std::vector<Tensor<1,3> > local_Bfield_curls (n_q_points);
  std::vector<typename dealii::internal::CurlType<3>::type> local_Bfield_curls (n_q_points);
  std::vector<Tensor<2,3> > local_Bfield_gradients (n_q_points);
  std::vector<double> local_ne_values (n_q_points);

  Tensor<1,3> temp1, temp2, temp3;

  temp_field = 0;
  temp_count.reinit(temp_field);
  temp_count = 0;
  double scale = 1e25;


  // Step 1
  // Get v = ∇ x B / n_e
  for (const auto &cell: dof_handler.active_cell_iterators())
    {
      // Reinit local
      fe_values.reinit (cell);
      cell->get_dof_indices (local_dof_indices);
      // Extract all local values
      fe_values.get_function_values (solution,
                                     local_solution_values);
      fe_values[Bfield].get_function_values (solution,
                                     local_Bfield_values);
      // fe_values[Bfield].get_function_gradients (solution,
      //                                local_Bfield_gradients);
      fe_values[Bfield].get_function_curls (solution,
                                     local_Bfield_curls);
      fe_values[ne].get_function_values (solution,
                                     local_ne_values);

      cell_matrix = 0;
      cell_rhs = 0;
      cell_test_solution = local_Bfield_curls;

      for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        {
          for (unsigned int j=0; j<dofs_per_cell; j=j+1)
          {
            // std::cout << fe_values.shape_value(j, q_index) * fe_values.JxW (q_index) << '\n';
            temp_field(local_dof_indices[j]) += local_ne_values[q_index] *
                                                fe_values.shape_value(j, q_index);
            temp_count(local_dof_indices[j]) += 1.0; //fe_values.shape_value(j, q_index);
          }
          for (unsigned int j=3; j<dofs_per_cell; j=j+4)
          {
            // std::cout << fe_values.shape_value(j, q_index) * fe_values.JxW (q_index) << '\n';
            temp_field(local_dof_indices[j]) += local_ne_values[q_index] *
                                                fe_values.shape_value(j, q_index);
            temp_count(local_dof_indices[j]) += fe_values.shape_value(j, q_index);
          }
        }
    }

    for (unsigned int i=0; i<temp_field.size(); ++i)
    {
      temp_field(i) /= temp_count(i);// / (dofs_per_cell / 4.);
      // TODO: This fix is not appropriate. A better solution should exist
    }

    test_solution = (temp_field);
    if (test_output)
    {
      fname = "test_copy.vtk";
      output_results_test (fname);
    }
}




void R2007::output_results (const unsigned int timestep_number) const
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

  const std::string filename =  "solution-" +
                                Utilities::int_to_string (timestep_number, 3) +
                                ".vtk";

  std::ofstream output (filename.c_str());
  data_out.write_vtk (output);

}


void R2007::output_results_test (std::string s) const
{
// Taken from step-20
  std::vector<std::string> solution_names(3, "u");
  solution_names.push_back ("p");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  interpretation (3,
                  DataComponentInterpretation::component_is_part_of_vector);
  interpretation.push_back (DataComponentInterpretation::component_is_scalar);
  DataOut<3> data_out;
  data_out.add_data_vector (dof_handler, test_solution, solution_names, interpretation);
  data_out.build_patches (1+1);
  std::ofstream output (s);
  data_out.write_vtk (output);

}



void R2007::propagate ()
{

}


void R2007::run ()
{

  const FEValuesExtractors::Vector Bfield (0);
  const FEValuesExtractors::Scalar ne (3);

  make_grid ();
  setup_system ();
  set_IC ();

  output_results (0);

  bool test_output = false;

  assemble_system (true);

  // assemble_system_test (true);

  double dt;
  dt=5e7;
  for (unsigned int j=1; j<50; j=j+1)
  {  std::cout << "Step: "
            << j
            << std::endl
            << dt*test_solution.block(0).mean_value()
            << std::endl
            << solution.block(0).mean_value()
            << std::endl;
    solution.add(dt, test_solution);
    output_results (j);
    assemble_system (test_output);
  }

}
