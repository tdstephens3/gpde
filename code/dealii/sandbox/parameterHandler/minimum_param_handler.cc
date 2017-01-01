// minimum_param_handler.cc

#include <deal.II/base/parameter_handler.h>


int main (int argc, char **argv)
{
  using namespace dealii;
  double a,b;
  int    j,k;

  ParameterHandler prm;
  
  // declare_parameters()
  prm.enter_subsection("test params");
    prm.declare_entry("a", "1.0", Patterns::Double(), "Test parameter a");
    prm.declare_entry("b", "2.0", Patterns::Double(), "Test parameter b");
    prm.declare_entry("j", "1", Patterns::Integer(), "Test parameter j");
    prm.declare_entry("k", "1", Patterns::Integer(), "Test parameter k");
  prm.leave_subsection();

  prm.read_input("test_params.prm");

  // parse_parameters()
  prm.enter_subsection("test params");
    a = prm.get_double("a");
    b = prm.get_double("b");
    j = prm.get_integer("j");
    k = prm.get_integer("k");
  prm.leave_subsection();
 
  printf("test parameters: doubles a,b: %0.2f, %0.2f\nints    j,k: %d, %d\n", a,b,j,k);
  

}
