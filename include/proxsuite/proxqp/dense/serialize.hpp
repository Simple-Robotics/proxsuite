//
// Copyright (c) 2022 INRIA
//
/**
 * @file serialize.hpp
 */

#ifndef PROXSUITE_DENSE_SERIALIZE_HPP
#define PROXSUITE_DENSE_SERIALIZE_HPP

#include <Eigen/Dense>
#include <cereal/cereal.hpp>

#include <fstream>
#include <string>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>

namespace cereal {
template<class Archive,
         class _Scalar,
         int _Rows,
         int _Cols,
         int _Options,
         int _MaxRows,
         int _MaxCols>
inline void
save(
  Archive& ar,
  Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> const& m)
{
  Eigen::Index rows = m.rows();
  Eigen::Index cols = m.cols();
  ar(rows);
  ar(cols);

  // TODO: remove assert and take storage order into account
  assert(m.IsColMajor);
  for (Eigen::Index i = 0; i < m.size(); i++)
    ar(m.data()[i]);
}

template<class Archive,
         class _Scalar,
         int _Rows,
         int _Cols,
         int _Options,
         int _MaxRows,
         int _MaxCols>
inline void
load(Archive& ar,
     Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& m)
{
  Eigen::Index rows;
  Eigen::Index cols;
  ar(rows);
  ar(cols);

  // TODO: remove assert and take storage order into account
  assert(m.IsColMajor);

  m.resize(rows, cols);

  for (Eigen::Index i = 0; i < m.size(); i++)
    ar(m.data()[i]);
}

template<class Archive, typename T>
void
serialize(Archive& archive, proxsuite::proxqp::dense::Model<T>& model)
{
  archive(CEREAL_NVP(model.dim),
          CEREAL_NVP(model.n_eq),
          CEREAL_NVP(model.n_in),
          CEREAL_NVP(model.n_total),
          CEREAL_NVP(model.H),
          CEREAL_NVP(model.g),
          CEREAL_NVP(model.A),
          CEREAL_NVP(model.b),
          CEREAL_NVP(model.C),
          CEREAL_NVP(model.l),
          CEREAL_NVP(model.u));
}

template<class Archive, typename T>
void
serialize(Archive& archive, proxsuite::proxqp::Settings<T>& settings)
{
  archive(CEREAL_NVP(settings.default_rho),
          CEREAL_NVP(settings.default_mu_eq),
          CEREAL_NVP(settings.default_mu_in),
          CEREAL_NVP(settings.alpha_bcl),
          CEREAL_NVP(settings.beta_bcl),
          CEREAL_NVP(settings.refactor_dual_feasibility_threshold),
          CEREAL_NVP(settings.refactor_rho_threshold),
          CEREAL_NVP(settings.mu_min_eq),
          CEREAL_NVP(settings.mu_min_in),
          CEREAL_NVP(settings.mu_max_eq_inv),
          CEREAL_NVP(settings.mu_update_factor),
          CEREAL_NVP(settings.mu_update_inv_factor),
          CEREAL_NVP(settings.cold_reset_mu_eq),
          CEREAL_NVP(settings.cold_reset_mu_in),
          CEREAL_NVP(settings.cold_reset_mu_eq_inv),
          CEREAL_NVP(settings.cold_reset_mu_in_inv),
          CEREAL_NVP(settings.eps_abs),
          CEREAL_NVP(settings.eps_rel),
          CEREAL_NVP(settings.max_iter),
          CEREAL_NVP(settings.max_iter_in),
          CEREAL_NVP(settings.safe_guard),
          CEREAL_NVP(settings.nb_iterative_refinement),
          CEREAL_NVP(settings.eps_refact),
          CEREAL_NVP(settings.verbose),
          CEREAL_NVP(settings.initial_guess),
          CEREAL_NVP(settings.update_preconditioner),
          CEREAL_NVP(settings.compute_preconditioner),
          CEREAL_NVP(settings.compute_timings),
          CEREAL_NVP(settings.preconditioner_max_iter),
          CEREAL_NVP(settings.preconditioner_accuracy),
          CEREAL_NVP(settings.eps_primal_inf),
          CEREAL_NVP(settings.eps_dual_inf),
          CEREAL_NVP(settings.bcl_update),
          CEREAL_NVP(settings.sparse_backend));
}

template<class Archive, typename T>
void
serialize(Archive& archive, proxsuite::proxqp::Info<T>& info)
{
  archive(CEREAL_NVP(info.mu_eq),
          CEREAL_NVP(info.mu_eq_inv),
          CEREAL_NVP(info.mu_in),
          CEREAL_NVP(info.mu_in_inv),
          CEREAL_NVP(info.rho),
          CEREAL_NVP(info.nu),
          CEREAL_NVP(info.iter),
          CEREAL_NVP(info.iter_ext),
          CEREAL_NVP(info.mu_updates),
          CEREAL_NVP(info.rho_updates),
          CEREAL_NVP(info.status),
          CEREAL_NVP(info.setup_time),
          CEREAL_NVP(info.solve_time),
          CEREAL_NVP(info.run_time),
          CEREAL_NVP(info.objValue),
          CEREAL_NVP(info.pri_res),
          CEREAL_NVP(info.dua_res),
          CEREAL_NVP(info.duality_gap),
          CEREAL_NVP(info.sparse_backend));
}

template<class Archive, typename T>
void
serialize(Archive& archive, proxsuite::proxqp::Results<T>& results)
{
  archive(CEREAL_NVP(results.x),
          CEREAL_NVP(results.y),
          CEREAL_NVP(results.z),
          // CEREAL_NVP(results.active_constraints)
          CEREAL_NVP(results.info));
}

} // namespace cereal

namespace proxsuite {
namespace serialization {

///
/// \brief Loads an object from a std::stringstream.
///
/// \tparam T Type of the object to deserialize.
///
/// \param[out] object Object in which the loaded data are copied.
/// \param[in]  is  string stream constaining the serialized content of the
/// object.
///
template<typename T>
inline void
loadFromStringStream(T& object, std::istringstream& is)
{
  cereal::JSONInputArchive ia(is);
  ia(object);
}

///
/// \brief Saves an object inside a std::stringstream.
///
/// \tparam T Type of the object to deserialize.
///
/// \param[in]   object Object in which the loaded data are copied.
/// \param[out]  ss String stream constaining the serialized content of the
/// object.
///
template<typename T>
inline void
saveToStringStream(const T& object, std::stringstream& ss)
{
  cereal::JSONOutputArchive oa(ss);
  oa(object);
}

///
/// \brief Loads an object from a std::string
///
/// \tparam T Type of the object to deserialize.
///
/// \param[out] object Object in which the loaded data are copied.
/// \param[in]  str  string constaining the serialized content of the object.
///
template<typename T>
inline void
loadFromString(T& object, const std::string& str)
{
  std::istringstream is(str);
  loadFromStringStream(object, is);
}

///
/// \brief Saves an object inside a std::string
///
/// \tparam T Type of the object to deserialize.
///
/// \param[in] object Object in which the loaded data are copied.
///
/// \returns a string  constaining the serialized content of the object.
///
template<typename T>
inline std::string
saveToString(const T& object)
{
  std::stringstream ss;
  saveToStringStream(object, ss);
  return ss.str();
}

///
/// \brief Loads an object from a binary file.
///
/// \tparam T Type of the object to deserialize.
///
/// \param[out] object Object in which the loaded data are copied.
/// \param[in] filename Name of the file containing the serialized data.
///
template<typename T>
inline void
loadFromBinary(T& object, const std::string& filename)
{
  std::ifstream ifs(filename.c_str(), std::ios::binary);
  if (ifs) {
    cereal::BinaryInputArchive ia(ifs);
    ia(object);
  } else {
    const std::string exception_message(filename +
                                        " does not seem to be a valid file.");
    throw std::invalid_argument(exception_message);
  }
}

///
/// \brief Saves an object inside a binary file.
///
/// \tparam T Type of the object to deserialize.
///
/// \param[in] object Object in which the loaded data are copied.
/// \param[in] filename Name of the file containing the serialized data.
///
template<typename T>
void
saveToBinary(const T& object, const std::string& filename)
{
  std::ofstream ofs(filename.c_str(), std::ios::binary);
  if (ofs) {
    cereal::BinaryOutputArchive oa(ofs);
    oa(object);
  } else {
    const std::string exception_message(filename +
                                        " does not seem to be a valid file.");
    throw std::invalid_argument(exception_message);
  }
}

///
/// \brief Loads an object from a JSON file.
///
/// \tparam T Type of the object to deserialize.
///
/// \param[out] object Object in which the loaded data are copied.
/// \param[in] filename Name of the file containing the serialized data.
///
template<typename T>
inline void
loadFromJSON(T& object, const std::string& filename)
{
  std::ifstream ifs(filename.c_str());
  if (ifs) {
    cereal::JSONInputArchive ia(ifs);
    ia(object);
  } else {
    const std::string exception_message(filename +
                                        " does not seem to be a valid file.");
    throw std::invalid_argument(exception_message);
  }
}

///
/// \brief Saves an object inside a JSON file.
///
/// \tparam T Type of the object to deserialize.
///
/// \param[in] object Object in which the loaded data are copied.
/// \param[in] filename Name of the file containing the serialized data.
///
template<typename T>
void
saveToJSON(const T& object, const std::string& filename)
{
  std::ofstream ofs(filename.c_str());
  if (ofs) {
    cereal::JSONOutputArchive oa(ofs);
    oa(object);
  } else {
    const std::string exception_message(filename +
                                        " does not seem to be a valid file.");
    throw std::invalid_argument(exception_message);
  }
}
}
}

#endif /* end of include guard PROXSUITE_DENSE_SERIALIZE_HPP */
