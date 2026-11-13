"""habitat_sim import 을 막는 Isaac 번들 numba 를 이 프로세스에서만 숨긴다.

문제:
  Isaac 번들 numba 0.59.1 + habitat 설치가 끌고 온 llvmlite 0.49.0 조합은 깨져 있다.
  numba 0.59 는 llvmlite 0.42 의 레거시 pass manager API(create_module_pass_manager 등)를
  쓰는데 0.49 에서 제거됐다. import 시점에 터지므로
  habitat_sim -> quaternion -> numba.njit 에서 habitat_sim 자체를 못 읽는다.

왜 숨기는 것으로 충분한가:
  numpy-quaternion 은 numba 가 없으면 순수 numpy 경로로 폴백한다 (가속만 빠짐).
  navmesh(PathFinder)는 habitat-sim 의 C++ 코드라 numba 와 무관하다.
  81 서버가 쓰는 건 PathFinder 뿐이다.

NAV_ALLOW_NUMBA=1 이면 숨기지 않는다 (numba 가 실제로 필요한 작업용 탈출구).
이미지·overlay 는 건드리지 않는다 (읽기 전용으로 쓸 수 있다).
"""
import os

if not os.environ.get('NAV_ALLOW_NUMBA'):
    import sys

    class _BlockNumba:
        _BLOCKED = ('numba', 'llvmlite')

        def find_module(self, fullname, path=None):
            return self if self._blocked(fullname) else None

        def find_spec(self, fullname, path=None, target=None):
            if self._blocked(fullname):
                raise ImportError(f'{fullname} is hidden by sitecustomize (numba/llvmlite mismatch)')
            return None

        def _blocked(self, fullname):
            root = fullname.split('.')[0]
            return root in self._BLOCKED

    sys.meta_path.insert(0, _BlockNumba())
