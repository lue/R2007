
using namespace dealii;


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
  const double c = 1.0; // 3e10; // cm/s
  const double e = 1.0; // electron charge

  // Model constatns
  const double n0 = 1.0; // Central electron density in cm^-3
  const double r0 = 1.0;  // Star radius 1e6 cm = 10 km
  const double B0 = 1.0;     // Magnitude of the torroidal field in G
  const double B1 = 1.0;     // Magnitude of the poloidal field in G (B0>>B1)
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

  Chi = c/4.0/numbers::PI/n/e/(r_cyl*r_cyl+1e-4*1e-4);
  Chi0 = c/4.0/numbers::PI/n0/e/(r0*r0+1e-4*1e-4);

  // Chi should always be positive by construction. This check is probably unnecesasry.
  if (Chi<0) Chi=-Chi;

  // Toroidal field magnitude
  B = B0*(Chi0*Chi0/Chi/Chi);

  // return values in order 3 components for B and one for n_e
  values(0) = B1;
  values(1) = B*std::sin(phi+numbers::PI/2.);
  values(2) = B*std::cos(phi+numbers::PI/2.);
  values(3) = n;
}


// namespace EquationData
// {
//   template <int dim>
//   class MultiComponentFunction: public Function<dim>
//   {
//   public:
//     MultiComponentFunction (const double initial_time = 0.);
//     void set_component (const unsigned int d);
//   protected:
//     unsigned int comp;
//   };
//   template <int dim>
//   MultiComponentFunction<dim>::
//   MultiComponentFunction (const double initial_time)
//     :
//     Function<dim> (1, initial_time), comp(0)
//   {}
//   template <int dim>
//   void MultiComponentFunction<dim>::set_component(const unsigned int d)
//   {
//     Assert (d<dim, ExcIndexRange (d, 0, dim));
//     comp = d;
//   }
//   template <int dim>
//   class Velocity : public MultiComponentFunction<dim>
//   {
//   public:
//     Velocity (const double initial_time = 0.0);
//     virtual double value (const Point<dim> &p,
//                           const unsigned int component = 0) const;
//     virtual void value_list (const std::vector< Point<dim> > &points,
//                              std::vector<double> &values,
//                              const unsigned int component = 0) const;
//   };
//   template <int dim>
//   Velocity<dim>::Velocity (const double initial_time)
//     :
//     MultiComponentFunction<dim> (initial_time)
//   {}
//   template <int dim>
//   void Velocity<dim>::value_list (const std::vector<Point<dim> > &points,
//                                   std::vector<double> &values,
//                                   const unsigned int) const
//   {
//     const unsigned int n_points = points.size();
//     Assert (values.size() == n_points,
//             ExcDimensionMismatch (values.size(), n_points));
//     for (unsigned int i=0; i<n_points; ++i)
//       values[i] = Velocity<dim>::value (points[i]);
//   }
//   template <int dim>
//   double Velocity<dim>::value (const Point<dim> &p,
//                                const unsigned int) const
//   {
//     if (this->comp == 0)
//       {
//         const double Um = 1.5;
//         const double H  = 4.1;
//         return 4.*Um*p(1)*(H - p(1))/(H*H);
//       }
//     else
//       return 0.;
//   }
//   template <int dim>
//   class Pressure: public Function<dim>
//   {
//   public:
//     Pressure (const double initial_time = 0.0);
//     virtual double value (const Point<dim> &p,
//                           const unsigned int component = 0) const;
//     virtual void value_list (const std::vector< Point<dim> > &points,
//                              std::vector<double> &values,
//                              const unsigned int component = 0) const;
//   };
//   template <int dim>
//   Pressure<dim>::Pressure (const double initial_time)
//     :
//     Function<dim> (1, initial_time)
//   {}
//   template <int dim>
//   double Pressure<dim>::value (const Point<dim> &p,
//                                const unsigned int) const
//   {
//     return 25.-p(0);
//   }
//   template <int dim>
//   void Pressure<dim>::value_list (const std::vector<Point<dim> > &points,
//                                   std::vector<double> &values,
//                                   const unsigned int) const
//   {
//     const unsigned int n_points = points.size();
//     Assert (values.size() == n_points, ExcDimensionMismatch (values.size(), n_points));
//     for (unsigned int i=0; i<n_points; ++i)
//       values[i] = Pressure<dim>::value (points[i]);
//   }
// }
