// QuickIK C++ bindings: include this header to get every type. Everything but
// this file is generated from the Rust bridge code in cpp/src/ (see
// cpp/build.rs).

#pragma once

#include <stdexcept>
#include <string>
#include <utility>

#include "quickik/Error.hpp"
#include "quickik/ErrorKind.hpp"
#include "quickik/ForwardKinematicsWorkspace.hpp"
#include "quickik/KeypointObservationList.hpp"
#include "quickik/KinematicTree.hpp"
#include "quickik/Mat3.hpp"
#include "quickik/ObservationSequence.hpp"
#include "quickik/OrthographicViewXY.hpp"
#include "quickik/PinholeCamera.hpp"
#include "quickik/Projection.hpp"
#include "quickik/Quat.hpp"
#include "quickik/SequentialSolver.hpp"
#include "quickik/SingleFrameSolver.hpp"
#include "quickik/SolverConfig.hpp"
#include "quickik/SolverResult.hpp"
#include "quickik/SolverResultIterator.hpp"
#include "quickik/SolverResultList.hpp"
#include "quickik/State.hpp"
#include "quickik/Vec2.hpp"
#include "quickik/Vec3.hpp"
#include "quickik/free_functions.hpp"

namespace quickik {

/// A `quickik::Error` thrown by `unwrap`.
class Exception : public std::runtime_error {
 public:
  Exception(ErrorKind kind, const std::string& message)
      : std::runtime_error(message), kind_(kind) {}

  ErrorKind kind() const { return kind_; }

 private:
  ErrorKind kind_;
};

/// The value of a fallible call's `result`, or throws its error as a
/// `quickik::Exception`. For example,
/// `auto tree = quickik::unwrap(quickik::KinematicTree::from_json_file(path));`
template <typename T>
T unwrap(diplomat::result<T, std::unique_ptr<Error>> result) {
  if (result.is_ok()) {
    return std::move(*std::move(result).ok());
  }
  std::unique_ptr<Error> error = *std::move(result).err();
  throw Exception(error->kind(), error->message());
}

}  // namespace quickik
