from pytocl.car import State, Command
from pytocl.driver import Driver
from pytocl.main import main


class SnakeOilDefaultDriver(Driver):
    def __init__(self):
        super().__init__()
        self.prev_command = Command()
        self.prev_command.accelerator = 1

    def drive(self, car_state: State) -> Command:

        command = Command()

        target_speed = 100

        # # Damage Control
        # target_speed -= car_state.damage * .05
        #
        # if target_speed < 25:
        #     target_speed = 25

        # # Steer To Corner
        # steer = car_state.angle * (10 / 3.14)
        # # Steer To Center
        # steer -= car_state.distance_from_center * .10
        #
        # if steer < -1:
        #     steer = -1
        # if steer > 1:
        #     steer = 1
        self.steer(car_state, 0.0, command)
        steer = command.steering

        if car_state.speed_x > target_speed:
            accelerate = 0.2
        else:
            accelerate = 0.8
            # Throttle Control
            if car_state.speed_x < target_speed - (steer * 50):
                accelerate += .01
            else:
                accelerate -= .01
            if car_state.speed_x < 10:
                accelerate += 1 / (car_state.speed_x + .1)

        wheelVel = car_state.wheel_velocities

        # Traction Control System
        if ((wheelVel[2] + wheelVel[3]) -
                (wheelVel[0] + wheelVel[1]) > 5):
            accelerate -= .2

        if accelerate < 0:
            accelerate = 0
        if accelerate > 1:
            accelerate = 1

        if accelerate > 0:
            if car_state.rpm > 8000:
                command.gear = car_state.gear + 1

        if car_state.rpm < 2500 and car_state.gear > 2:
            command.gear = car_state.gear - 1

        if not command.gear:
            command.gear = car_state.gear or 1

        command.accelerator = accelerate
        command.steering = steer
        print(accelerate)
        self.prev_command = command

        return command


class LanqDriver(Driver):
    def __init__(self):
        super().__init__()

        self.steer_lock = 0.785398
        self.max_speed = 100
        self.prev_rpm = None
        self.prev_accel = None
        self.angles = [0 for x in range(19)]

        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15

        for i in range(5, 9):
            self.angles[i] = -20 + (i - 5) * 5
            self.angles[18 - i] = 20 - (i - 5) * 5

    def drive(self, carstate: State):
        # https://github.com/lanquarden/pyScrcClient/blob/master/src/driver.py
        steer = (carstate.angle - carstate.distance_from_center * 0.5) / self.steer_lock
        rpm = carstate.rpm
        gear = carstate.gear

        if self.prev_rpm is None:
            up = True
        else:
            if (self.prev_rpm - rpm) < 0:
                up = True
            else:
                up = False

        if up and rpm > 7000:
            gear += 1

        if not up and rpm < 3000:
            gear -= 1

        speed = carstate.speed_x
        accel = 1 if self.prev_accel is None else self.prev_accel

        if speed < self.max_speed:
            accel += 0.1
            if accel > 1:
                accel = 1.0
        else:
            accel -= 0.1
            if accel < 0:
                accel = 0.0

        command = Command()
        command.accelerator = accel
        command.steering = steer
        command.gear = gear

        self.prev_accel = command.accelerator

        return command


if __name__ == '__main__':
    main(LanqDriver())
