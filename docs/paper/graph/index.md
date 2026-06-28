이 그래프는 MicroLad 논문을 섹션별 요약이 아니라 주장과 근거의 지도로 정리한다.

## 읽는 순서

현재 만든 노드는 여기까지다.

1. [[docs/paper/graph/background|배경]]: 이 논문이 필요한 직접적인 상황.
2. [[docs/paper/graph/claim|논문의 주장]]: 논문의 중심 주장.
3. [[docs/paper/graph/method|방법]]: VAE, LDM, L-MPDD, SDS가 어떻게 이어지는지.
4. [[docs/paper/graph/reconstruction|재구성]]: 2D 데이터로 패턴이 맞는 3D 후보를 만드는 근거.
5. [[docs/paper/graph/control|역방향 제어]]: 3D 후보를 만든 뒤 목표 수치 쪽으로 조정하는 후속 단계.
6. [[docs/paper/graph/limits|한계]]: 가정, 빈틈, 비용, 조심해서 읽어야 하는 부분.

기본 읽기 순서는 [[docs/paper/graph/background|배경]]에서 시작한다.

## 보조 개념

- [[docs/paper/graph/mpdd|MPDD]]: 3D 후보를 슬라이스해서 보고 고치는 방식. SliceGAN과 헷갈릴 때 읽는다.
- [[docs/paper/graph/s2|S2]]: 같은 물질이 거리별로 얼마나 같이 나타나는지 보는 패턴 수치. 덩어리 크기나 연결성과 헷갈릴 때 읽는다.

원문은 각 노드의 숫자 레퍼런스가 가리키는 근거다.

- [추출한 논문 원문과 그림](../md/2508.20138v4.md)

## 작성 기준

각 노드는 자기 내용만으로 읽혀야 한다.

문체는 [[docs/paper/graph/background|배경]]과 [[docs/paper/graph/claim|논문의 주장]]의 흐름을 유지한다.

- 앞에서 쓰던 핵심어를 이어 쓴다: 2D 조각 이미지, 3D 후보, 목표 수치, 패턴 수치.
- 첫 문장은 그 노드의 중심 내용을 바로 말한다.
- 전문용어는 먼저 던지지 않고, 쉬운 역할 설명 뒤에 붙인다.
- 링크는 현재 문장의 근거와 연결로 쓴다.
