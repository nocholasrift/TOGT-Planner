#include "drolib/polynomial/arc_len_traj_converter.hpp"
#include "drolib/race/race_params.hpp"
#include "drolib/race/race_planner.hpp"
#include "drolib/race/race_track.hpp"
#include <filesystem>
using namespace drolib;

int main() {
  const std::string quad_name = "crazyflie";
  const std::string config_name = quad_name + "_setups.yaml";
  // const std::string track_name = "race_uzh_7g_multiprisma.yaml";
  const std::string track_name = "straight_line.yaml";
  // const std::string track_name = "race_uzh_19wp.yaml";

  const std::string traj_name = "crazyflie_traj.csv";
  const std::string arclen_traj_name = "crazyflie_arclen_traj.csv";
  const std::string wpt_name = "crazyflie_wpt.yaml";

  std::shared_ptr<RaceTrack> racetrack;
  std::shared_ptr<RacePlanner> raceplanner;
  std::shared_ptr<RaceParams> raceparams;

  fs::path root(PROJECT_ROOT);
  fs::path config_path = root / "parameters" / quad_name;
  fs::path track_path = root / "resources/racetrack" / track_name;
  fs::path traj_path = root / "resources/trajectory" / traj_name;
  fs::path arclen_traj_path = root / "resources/trajectory" / arclen_traj_name;
  fs::path wpt_path = root / "resources/trajectory" / wpt_name;

  raceparams = std::make_shared<RaceParams>(config_path, config_name);
  raceplanner = std::make_shared<RacePlanner>(*raceparams);

  racetrack = std::make_shared<RaceTrack>(track_path);
  bool status = raceplanner->planTOGT(racetrack);

  if (!status) {
    std::cout << "planner failed.\n";
    return -1;
  }

  TrajExtremum extremum = raceplanner->getExtremum();
  std::cout << extremum << std::endl;

  MincoSnapTrajectory traj = raceplanner->getTrajectory();
  traj.save(traj_path);
  traj.saveSegments(wpt_path, 1);

  ArclenTrajConverter::Settings converter_settings;
  converter_settings.num_samples = 100;
  converter_settings.arclen_resolution = 2000;
  converter_settings.min_sample_dist = 100;

  ArclenTrajConverter arclen_converter(converter_settings);
  arclen_converter.convert(traj);
  arclen_converter.save(arclen_traj_path);

  std::cout << "trajectory saved to " << traj_path << "\n";
  std::cout << "arclen trajectory saved to " << arclen_traj_path << "\n";
  std::cout << "wpts saved to " << wpt_path << "\n";

  return 0;
}
