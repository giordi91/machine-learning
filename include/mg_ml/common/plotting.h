#pragma once
#include <vector>
#include <string>
#include <sstream>

/*This is a simple thin wrapper around gnuplot to try to visualize some of the
 *data, there is no fancy magic going on here, in the end a system call will be
 *made with the composed plotting command
 */
namespace plot {

const std::string SINGLE_QUOTE{"'",1};

struct GnuOption {
  std::string name;
  std::string value;
};

struct GnuFile {
public:
  void toString(std::ostringstream &oss) const {
    if (add_quotes) {
      oss << SINGLE_QUOTE << name << SINGLE_QUOTE;
    } else {
      oss << name;
    }
    for(const auto& o : options)
    {
        oss<<" "<<o.name<<" "<<o.value<<" ";
    
    }
  }

public:
  std::vector<GnuOption> options;
  std::string name;
  bool add_quotes = true;
};

struct GnuPlot {

  explicit GnuPlot() = default;
  void show() const;

  std::string name;
  std::vector<GnuFile> files;
};

void GnuPlot::show() const {
  std::ostringstream oss;
  //initializing the plot command and adding title
  oss << "gnuplot -e \" "<< "set title "<<SINGLE_QUOTE<<name<<SINGLE_QUOTE<<"; "<< "plot ";
  for (const auto &f : files) {
    f.toString(oss);
    oss << ", ";
  }

  oss.seekp(-2, oss.cur);
  oss << "; pause -1 ;\"";
  //,f(x) = 2 + 3*x, f(x)
  const std::string outs = oss.str();
  std::cout << outs << std::endl;
  system(outs.c_str());
}

inline GnuFile plot_line(float constant, float slope) {
  GnuFile f;
  f.add_quotes = false;
  f.name = std::string("f(x) = ") + std::to_string(constant) + " +  " +
           std::to_string(slope) + "*x, f(x)";
  return f;
};

inline GnuFile plot_image(const std::string& path)
{
  GnuFile f;
  f.name = path;
  f.options.emplace_back(GnuOption{"with","rgbimage"});
  return f;
}
}//end namespace plot
