#include <doctest.hpp>

#include <proxsuite/proxqp/serialization/archive.hpp>
#include <proxsuite/proxqp/serialization/eigen.hpp>
#include <proxsuite/proxqp/serialization/model.hpp>
#include <proxsuite/proxqp/serialization/results.hpp>
#include <proxsuite/proxqp/serialization/settings.hpp>

template<typename object>
struct init;

template<typename T>
struct init<proxsuite::proxqp::dense::Model<T>>
{
  typedef proxsuite::proxqp::dense::Model<T> Model;

  static Model run()
  {
    Model model(1, 0, 0);
    return model;
  }
};

template<typename T>
struct init<proxsuite::proxqp::Results<T>>
{
  typedef proxsuite::proxqp::Results<T> Results;

  static Results run()
  {
    Results results;
    return results;
  }
};

template<typename T>
struct init<proxsuite::proxqp::Settings<T>>
{
  typedef proxsuite::proxqp::Settings<T> Settings;

  static Settings run()
  {
    Settings settings;
    return settings;
  }
};

template<typename T>
void
generic_test(const T& object, const std::string& filename)
{
  using namespace proxsuite::serialization;

  // Load and save as XML
  const std::string xml_filename = filename + ".xml";
  saveToXML(object, xml_filename);

  {
    T object_loaded = init<T>::run();
    loadFromXML(object_loaded, xml_filename);

    // Check
    DOCTEST_CHECK(object_loaded == object);
  }

  // Load and save as json
  const std::string json_filename = filename + ".json";
  saveToJSON(object, json_filename);

  {
    T object_loaded = init<T>::run();
    loadFromJSON(object_loaded, json_filename);

    // Check
    DOCTEST_CHECK(object_loaded == object);
  }

  // Load and save as binary
  const std::string bin_filename = filename + ".bin";
  saveToBinary(object, bin_filename);

  {
    T object_loaded = init<T>::run();
    loadFromBinary(object_loaded, bin_filename);

    // Check
    DOCTEST_CHECK(object_loaded == object);
  }
}