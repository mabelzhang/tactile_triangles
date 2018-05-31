// Mabel Zhang
// 29 Dec 2016
//
// Refactored from sample_pcl.cpp, to be used by active_visual_tactile
//   package.
// Reads config/resize_3dnet_cat10_train.csv to get a map between 3DNet
//   classname and the scaling factor to use so that object model is of
//   real-world size.
//

// C++
#include <fstream>  // for ifstream


// Read config file specifying how much to scale down the point clouds in each
//   3DNet class.
void read_scaling_config (std::string file_path,
  std::map <std::string, float> & rescale_map)
{
  std::vector <std::string> keys;
  std::vector <float> values;

  std::ifstream infile (file_path);

  int line_count = 0;

  std::string line;
  std::string field;

  // Read 1st line. This is the header (column titles) line. These are keys
  //   of map.
  infile >> line;
  std::istringstream iss1 (line);
  // Ref: http://stackoverflow.com/questions/19936483/c-reading-csv-file
  while (std::getline (iss1, field, ','))
    keys.push_back (field);

  // Read 2nd line. These are the values of map
  infile >> line;
  std::istringstream iss2 (line);
  while (std::getline (iss2, field, ','))
    values.push_back (atof (field.c_str ()));

  // Make sure config file is valid, i.e. number of elts in each row is same!
  assert (keys.size () == values.size ());

  for (int i = 0; i < keys.size (); i ++)
  {
    rescale_map.insert (std::pair <std::string, float> (keys [i], values [i]));
    fprintf (stderr, "Rescaling %s by %f\n", keys[i].c_str (), values[i]);
  }
}
